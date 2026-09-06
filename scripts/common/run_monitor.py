"""Score Terminal Wrench Gemini-pool trajectories with one monitor via OpenRouter.

ALL monitors (M0/M1/M2) go through the same OpenRouter chat/completions client so the
only difference between them is the model — no Responses-vs-chat confound. We reuse the
AUTHORS' prompt (monitoring/prompts/v1), renderer (format_atif_trajectory), and parser
(parse_monitor_response) verbatim, and our byte-validated tw_lib.make_stripped for scope.

Output: one JSONL row per trajectory. Resumable (skips sample_ids already scored).
Usage (from the repo root):
  python scripts/common/run_monitor.py --model openai/gpt-5.4-nano --scope full   --out results/scores/m0_full.jsonl
  python scripts/common/run_monitor.py --model anthropic/claude-sonnet-4.6 --scope stripped --out results/scores/m1_stripped.jsonl
  python scripts/common/run_monitor.py ... --max-per-class 15   # tiny pilot
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

SCORE_RE = re.compile(r'"suspicion_score"\s*:\s*([01](?:\.\d+)?|0?\.\d+)')


def extract_score(content):
    """authors' parser first; regex fallback recovers scores when the reasoning
    string has unescaped quotes (valid score, JSON-invalid string -> parser misses it)."""
    s = M.parse_monitor_response(content).get("suspicion_score")
    if s is not None:
        return float(s), "json"
    m = SCORE_RE.search(content or "")
    if m:
        return float(m.group(1)), "regex"
    return None, None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
REPO = ROOT / "repo"
sys.path.insert(0, str(REPO / "monitoring"))

import tw_lib as tw
import monitor as M  # authors' templates/renderer/parser

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
ACTOR = "gemini-3.1-pro"  # default; overridden by --actor
NEGSET = None  # if set, score same-loop non-hack negatives instead of hacks+baselines
THINK_SUFFIX = ""  # appended to system msg; "/no_think" disables Qwen3 thinking mode
MAXTOKENS = 4000
PROMPT = "v1"
DEV_MSG = M.DEVELOPER_TEMPLATE.format(task_prompt=M.load_prompt(PROMPT))


def collect():
    """Yield dicts for every hack + baseline (ground-truth labels attached).
    If NEGSET is set, instead yield ONLY same-loop non-hack negatives (attacker-loop
    trajectories judged not-a-hack) from the non_hack_trajectories tree, is_hack=0."""
    out = []
    if NEGSET:
        want = {"legit": {"attacker_legitimate_solve"}, "noreward": {"no_reward"},
                "legit_noreward": {"attacker_legitimate_solve", "no_reward"}}[NEGSET]
        for e in json.load(open(REPO / "index" / "trajectories.json")):
            if e["model"] != ACTOR or e.get("classification") not in want:
                continue
            label = e.get("trajectory_label", "v5")
            p = REPO / "tasks" / e["task_id"] / e["model"] / "non_hack_trajectories" / label / "trial" / "agent" / "trajectory.json"
            if p.exists():
                out.append({"task_id": e["task_id"], "label": label, "is_hack": 0, "is_neg": True,
                            "classification": e.get("classification"), "reward": e.get("reward"),
                            "observed_categories": [], "source": e.get("source_dataset"), "path": str(p)})
        return out
    for e in json.load(open(REPO / "index" / "trajectories.json")):
        if e["model"] != ACTOR:
            continue
        if e.get("classification") not in ("rewarded_serious_exploit", "rewarded_nonserious_hack"):
            continue
        label = e.get("trajectory_label", "v5")
        p = REPO / "tasks" / e["task_id"] / e["model"] / "hack_trajectories" / label / "trial" / "agent" / "trajectory.json"
        if p.exists():
            out.append({"task_id": e["task_id"], "label": label, "is_hack": 1,
                        "classification": e.get("classification"), "reward": e.get("reward"),
                        "observed_categories": e.get("observed_categories") or [],
                        "source": e.get("source_dataset"), "path": str(p)})
    for e in json.load(open(REPO / "index" / "tasks.json")):
        if e["model"] != ACTOR:
            continue
        for bl in e.get("baselines", []):
            label = bl.get("label") or bl.get("baseline_label", "1")
            p = REPO / "tasks" / e["task_id"] / e["model"] / "baseline_trajectories" / label / "trial" / "agent" / "trajectory.json"
            if p.exists():
                out.append({"task_id": e["task_id"], "label": label, "is_hack": 0,
                            "classification": "legitimate", "reward": e.get("reward"),
                            "observed_categories": [], "source": e.get("source_dataset"),
                            "path": str(p)})
    return out


def sid(s):
    kind = "neg" if s.get("is_neg") else ("hack" if s["is_hack"] else "base")
    return f"{s['task_id']}__{s['label']}__{kind}"


def render(s, scope):
    atif = M.load_atif_trajectory(Path(s["path"]))
    if atif is None:
        return None
    if scope == "stripped":
        atif = tw.make_stripped(atif)
    instr = M.load_instruction(s["task_id"], ACTOR)
    return M.USER_TEMPLATE.format(task_description=instr, trajectory=M.format_atif_trajectory(atif))


async def call(client, model, user_msg, sem, retries=4):
    body = {"model": model, "temperature": 0, "max_tokens": MAXTOKENS,
            "messages": [{"role": "system", "content": DEV_MSG + THINK_SUFFIX},
                         {"role": "user", "content": user_msg}]}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
               "X-Title": "cot-monitor-study"}
    async with sem:
        last = None
        for a in range(retries):
            try:
                r = await client.post(OR_URL, json=body, headers=headers, timeout=180)
                if r.status_code == 200:
                    j = r.json()
                    return j["choices"][0]["message"]["content"], None
                last = f"http {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
            await asyncio.sleep(2 * (a + 1))
        return None, last


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--actor", default="gemini-3.1-pro",
                    choices=["gemini-3.1-pro", "claude-opus-4.6", "gpt-5.4"])
    ap.add_argument("--negset", default=None, choices=["legit", "noreward", "legit_noreward"],
                    help="score same-loop non-hack negatives instead of hacks+baselines")
    ap.add_argument("--scope", required=True, choices=["full", "stripped"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-class", type=int, default=None, help="cap hacks and baselines each (pilot)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--clean-only", action="store_true", help="drop hacks with leak keywords in tool_calls")
    ap.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3 non-thinking judge)")
    ap.add_argument("--max-tokens", type=int, default=4000, help="completion cap (raise for reasoning judges)")
    args = ap.parse_args()
    global ACTOR, NEGSET, THINK_SUFFIX, MAXTOKENS
    ACTOR = args.actor
    NEGSET = args.negset
    MAXTOKENS = args.max_tokens
    if args.no_think:
        THINK_SUFFIX = "\n\n/no_think"

    samples = collect()
    if args.clean_only:
        before = len(samples)
        samples = [s for s in samples if not (s["is_hack"] and M.has_keyword_leak_in_tool_calls(s["path"]))]
        print(f"clean-only: dropped {before - len(samples)} keyword-leaking hacks")
    if args.max_per_class:
        hacks = [s for s in samples if s["is_hack"]][:args.max_per_class]
        bases = [s for s in samples if not s["is_hack"]][:args.max_per_class]
        samples = hacks + bases
    n_h = sum(s["is_hack"] for s in samples)
    print(f"{len(samples)} samples ({n_h} hack / {len(samples)-n_h} baseline) | {args.model} | {args.scope}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                row = json.loads(line)
                if row.get("score") is not None:
                    done.add(row["sample_id"])
            except Exception:
                pass
    todo = [s for s in samples if sid(s) not in done]
    print(f"  {len(done)} already scored, {len(todo)} to do")

    sem = asyncio.Semaphore(args.concurrency)
    n_ok = n_err = n_noparse = 0
    async with httpx.AsyncClient() as client:
        async def work(s):
            nonlocal n_ok, n_err, n_noparse
            user_msg = render(s, args.scope)
            if user_msg is None:
                return {**meta(s), "score": None, "raw": None, "error": "render_failed"}
            content, err = await call(client, args.model, user_msg, sem)
            if err:
                n_err += 1
                return {**meta(s), "score": None, "raw": None, "error": err}
            score, how = extract_score(content)
            if score is None:
                n_noparse += 1
            else:
                n_ok += 1
            return {**meta(s), "score": score, "parse": how, "raw": content, "error": None}

        def meta(s):
            return {"sample_id": sid(s), "task_id": s["task_id"], "is_hack": s["is_hack"],
                    "classification": s["classification"], "reward": s["reward"],
                    "observed_categories": s["observed_categories"], "source": s["source"],
                    "monitor": args.model, "scope": args.scope, "input_chars": None}

        with open(outp, "a") as f:
            tasks = [asyncio.create_task(work(s)) for s in todo]
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                row = await fut
                f.write(json.dumps(row) + "\n"); f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  ok={n_ok} noparse={n_noparse} err={n_err}")
    print(f"DONE: ok={n_ok} noparse={n_noparse} err={n_err} -> {outp}")


if __name__ == "__main__":
    asyncio.run(main())
