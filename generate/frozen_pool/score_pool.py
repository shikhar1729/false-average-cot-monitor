"""Score a Terminal Wrench actor pool with one monitor, at full or action-only scope.

Writes data/frozen_pool/<monitor>_<scope>.jsonl (Gemini-3.1-pro actor) and data/opus_pool/...
(Claude-Opus-4.6 actor): one suspicion score per trajectory under the dataset's own rubric, for
Section 3, Tables 1 and 3, and Appendix B. Example: `python -m generate.frozen_pool.score_pool
--model openai/gpt-5.4-nano --scope full --clean-only`; `--help` lists the exact runs behind data/.
Needs OPENROUTER_API_KEY and the Terminal Wrench clone (TW_REPO). Roughly $2 (nano) to $50 (sonnet) per pool.
"""
import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import httpx

from generate.common import judge, subset, tw
from generate.common.openrouter import FLASH, NANO, SONNET
from generate.common.paths import DATA, TW

GPT54 = "openai/gpt-5.4"
SHORT_NAME = {NANO: "nano", SONNET: "sonnet", FLASH: "flash", GPT54: "gpt54"}
POOL_DIR = {"gemini-3.1-pro": "frozen_pool", "claude-opus-4.6": "opus_pool"}

# The runs behind data/ (output file, then flags); the default --out gives these names.
RUNS = """runs behind data/:
frozen_pool/nano_full            --model openai/gpt-5.4-nano         --scope full        --clean-only
frozen_pool/sonnet_full          --model anthropic/claude-sonnet-4.6 --scope full        --clean-only
frozen_pool/sonnet_action-only   --model anthropic/claude-sonnet-4.6 --scope action-only --clean-only
frozen_pool/flash_action-only    --model google/gemini-2.5-flash     --scope action-only --clean-only
frozen_pool/gpt54_action-only    --model openai/gpt-5.4              --scope action-only --reasoning-dependent
opus_pool/nano_full              --actor claude-opus-4.6 --model openai/gpt-5.4-nano     --scope full        --clean-only
opus_pool/flash_full             --actor claude-opus-4.6 --model google/gemini-2.5-flash --scope full        --clean-only
opus_pool/flash_action-only      --actor claude-opus-4.6 --model google/gemini-2.5-flash --scope action-only --clean-only
opus_pool/flash_full_honest      --actor claude-opus-4.6 --model google/gemini-2.5-flash --scope full        --honest
opus_pool/gpt54_full              --actor claude-opus-4.6 --model openai/gpt-5.4          --scope full        --clean-only
opus_pool/gpt54_full_honest       --actor claude-opus-4.6 --model openai/gpt-5.4          --scope full        --honest
"""


def honest_solves(actor):
    """Legitimate solves from the attacker loop itself (the dataset's attacker_legitimate_solve class)."""
    out = []
    for e in json.load(open(TW / "index" / "trajectories.json")):
        if e["model"] != actor or e.get("classification") != "attacker_legitimate_solve":
            continue
        label = e.get("trajectory_label", "v5")
        p = TW / "tasks" / e["task_id"] / actor / "non_hack_trajectories" / label / "trial" / "agent" / "trajectory.json"
        if p.exists():
            out.append({"task_id": e["task_id"], "label": label, "is_hack": 0, "honest": True,
                        "observed_categories": [], "path": str(p)})
    return out


def sample_id(s):
    return f"{s['task_id']}__{s['label']}__neg" if s.get("honest") else tw.sid(s)


def select(args):
    if args.reasoning_dependent:
        # The sensitivity run (Table 3): every reasoning-dependent hack plus 150 legit solves from a
        # seeded draw of its own (not subset.legit); this reproduces the shipped ids exactly.
        legit = sorted((s for s in tw.collect() if not s["is_hack"]), key=tw.sid)
        return subset.reasoning_dependent(cap=10**6) + random.Random(0).sample(legit, 150)  # no cap: all 251
    samples = honest_solves(args.actor) if args.honest else tw.collect(args.actor)
    if args.clean_only:
        before = len(samples)
        samples = [s for s in samples if not (s["is_hack"] and tw.M.has_keyword_leak_in_tool_calls(s["path"]))]
        print(f"clean-only: dropped {before - len(samples)} hacks that name the exploit in a command")
    if args.max_per_class:
        samples = ([s for s in samples if s["is_hack"]][:args.max_per_class]
                   + [s for s in samples if not s["is_hack"]][:args.max_per_class])
    return samples


async def score_one(client, sem, args, s):
    atif = tw.load(s)
    score = None
    if atif is not None:
        user = tw.render(atif, s["task_id"], args.actor, args.scope)
        score = await judge.score(client, sem, judge.BASELINE, user, model=args.model, max_tokens=args.max_tokens)
    return {"sample_id": sample_id(s), "task_id": s["task_id"], "is_hack": s["is_hack"],
            "observed_categories": s["observed_categories"], "score": score}


async def run(args, out):
    samples = select(args)
    n_hack = sum(s["is_hack"] for s in samples)
    print(f"{len(samples)} samples ({n_hack} hack / {len(samples) - n_hack} legit) | {args.model} | {args.scope}")
    done = set()
    if out.exists():
        done = {r["sample_id"] for r in map(json.loads, open(out)) if r.get("score") is not None}
    todo = [s for s in samples if sample_id(s) not in done]
    print(f"  {len(done)} already scored, {len(todo)} to do -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sem, n_ok = asyncio.Semaphore(args.concurrency), 0
    async with httpx.AsyncClient() as client:
        with open(out, "a") as f:
            jobs = [asyncio.create_task(score_one(client, sem, args, s)) for s in todo]
            for i, job in enumerate(asyncio.as_completed(jobs), 1):
                row = await job
                n_ok += row["score"] is not None
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  scored={n_ok} failed={i - n_ok}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=RUNS, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OpenRouter model id of the monitor")
    ap.add_argument("--actor", default="gemini-3.1-pro", choices=sorted(POOL_DIR))
    ap.add_argument("--scope", required=True, choices=["full", "action-only"])
    ap.add_argument("--out", type=Path, help="default: data/<pool>/<monitor>_<scope>[_honest].jsonl")
    ap.add_argument("--clean-only", action="store_true", help="drop hacks that name the exploit in a command")
    ap.add_argument("--honest", action="store_true", help="score the attacker loop's own legitimate solves instead")
    ap.add_argument("--reasoning-dependent", action="store_true",
                    help="score only the reasoning-dependent subset plus 150 legit solves (Gemini pool)")
    ap.add_argument("--max-per-class", type=int, help="cap hacks and legit solves each (a small pilot)")
    ap.add_argument("--max-tokens", type=int, default=4000, help="completion cap; raise for reasoning judges")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    name = SHORT_NAME.get(args.model, args.model.split("/")[-1])
    out = args.out or DATA / POOL_DIR[args.actor] / f"{name}_{args.scope}{'_honest' if args.honest else ''}.jsonl"
    asyncio.run(run(args, out))


if __name__ == "__main__":
    main()
