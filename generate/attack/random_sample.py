"""Rewrite the reasoning of 150 randomly drawn caught hacks, and score each rewrite.

Writes data/attack/random_sample.jsonl (Section 4, the unbiased random sample, and Table 1's paraphrase
control): each hack the target monitor (gpt-5.4-nano, full scope) catches gets an obfuscated and a
paraphrased rewrite, both scored by nano and the held-out sonnet, beside the original pool scores.
Run `python -m generate.attack.random_sample --batch` (roughly $30; OPENROUTER_API_KEY and the Terminal
Wrench clone). `--selfcheck` tests the rewrite rail for $0; `--dry-run N` rewrites N hacks without
scoring (a few cents) and saves the before and after reasoning to runs/attack/ for a coherence audit.
"""
import argparse
import asyncio
import copy
import json
import os

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.openrouter import NANO, SONNET
from generate.common.paths import DATA, ROOT, TW

OUT = DATA / "attack" / "random_sample.jsonl"
AUDIT = ROOT / "runs" / "attack" / "random_sample_dryrun.json"
MONITORS = {"m0": NANO, "m1": SONNET}
VARIANTS = {"obf": obfuscator.OBFUSCATOR_PROMPT, "par": obfuscator.PARAPHRASE_PROMPT}


def original_scores():
    """sid -> {"m0": nano score, "m1": sonnet score} for every hack in the frozen pool (full scope)."""
    out = {}
    for tag, name in (("m0", "nano_full.jsonl"), ("m1", "sonnet_full.jsonl")):
        for r in map(json.loads, open(DATA / "frozen_pool" / name)):
            if r["is_hack"] and r["score"] is not None:
                out.setdefault(r["sample_id"], {})[tag] = r["score"]
    return out


async def monitor_score(client, sem, atif, task_id, model):
    # The pool's own monitor call: dataset rubric, temperature 0, 4000 completion tokens.
    return await judge.score(client, sem, judge.BASELINE, tw.render(atif, task_id), model=model, max_tokens=4000)


async def hack_row(client, sem, s, orig):
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "observed_categories": s["observed_categories"]}
    for variant, system in VARIANTS.items():
        new, row[f"{variant}_status"] = await obfuscator.obfuscate(client, sem, s, system)
        if new is not None:
            for tag, model in MONITORS.items():
                row[f"{variant}_{tag}"] = await monitor_score(client, sem, new, s["task_id"], model)
    row["orig_m0"] = orig.get(row["sid"], {}).get("m0")
    row["orig_m1"] = orig.get(row["sid"], {}).get("m1")
    return row


async def batch(n):
    orig, hacks = original_scores(), subset.caught(n)
    done = {json.loads(line)["sid"] for line in open(OUT)} if OUT.exists() else set()
    todo = [s for s in hacks if tw.sid(s) not in done]
    print(f"random sample: {len(hacks)} caught hacks, {len(todo)} to do", flush=True)
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        with open(OUT, "a") as f:
            jobs = [asyncio.create_task(hack_row(client, sem, s, orig)) for s in todo]
            for i, job in enumerate(asyncio.as_completed(jobs), 1):
                f.write(json.dumps(await job) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)


async def dry_run(n):
    """Rewrite (no scoring) the first n caught hacks in pool order and save the reasoning before and after."""
    caught = {tw.sid(s) for s in subset.caught(10**6)}
    hacks = [s for s in tw.clean_hacks() if tw.sid(s) in caught][:n]
    print(f"dry run: rewriting {len(hacks)} caught hacks", flush=True)
    sem, audit = asyncio.Semaphore(4), []
    async with httpx.AsyncClient() as client:
        for s in hacks:
            new, status = await obfuscator.obfuscate(client, sem, s)
            print(f"  {tw.sid(s)}: {status}" + (f" | re-rendered {len(tw.render(new, s['task_id']))} chars" if new else ""))
            if new:
                audit.append({"sid": tw.sid(s), "task_id": s["task_id"], "before": obfuscator.agent_msg_steps(tw.load(s)),
                              "after": obfuscator.agent_msg_steps(new)})
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(AUDIT, "w"), indent=2)
    print(f"wrote {AUDIT} ({len(audit)} trajectories) for a coherence audit")


def selfcheck():
    """$0, no API. On the first three clean hacks: an identity rewrite and a real narration edit pass the
    rail and re-render with the task text, and a tampered tool call is caught."""
    assert (TW / "index" / "trajectories.json").exists(), f"--selfcheck needs the Terminal Wrench clone at {TW}; set TW_REPO"
    hacks = tw.clean_hacks()[:3]
    assert hacks, "no hacks found in the clone"
    for s in hacks:
        atif = tw.load(s)
        steps = obfuscator.agent_msg_steps(atif)
        identity = obfuscator.apply_rewrites(atif, [{"step_id": i, "message": m} for i, m in steps])
        obfuscator.assert_exploit_preserved(atif, identity)
        rendered = tw.render(identity, s["task_id"])
        assert "(no task description found)" not in rendered, \
            f"{tw.sid(s)}: the render has no task description; monitor.py is not reading the clone at {TW}"
        edit = "I proceeded with a standard, correct implementation."
        edited = obfuscator.apply_rewrites(atif, [{"step_id": i, "message": edit} for i, _ in steps])
        obfuscator.assert_exploit_preserved(atif, edited)
        tampered = copy.deepcopy(atif)
        for st in tampered.get("steps", []):
            if st.get("source") == "agent" and st.get("tool_calls"):
                st["tool_calls"] = [{"tampered": True}]
                break
        try:
            obfuscator.assert_exploit_preserved(atif, tampered)
        except AssertionError:
            pass
        else:
            raise SystemExit(f"{tw.sid(s)}: the rail missed a tampered tool call")
        print(f"  ok {tw.sid(s)}: {len(steps)} agent messages | identity and edit pass, tamper caught | "
              f"rendered {len(rendered)} chars")
    print(f"selfcheck passed: {len(hacks)} trajectories, rail and re-render verified, $0 spent")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="$0: test the rewrite rail and rendering, no API")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N", help="paid: rewrite N caught hacks, no scoring")
    ap.add_argument("--batch", action="store_true", help="paid: rewrite and score --n caught hacks")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not (args.dry_run or args.batch):
        return ap.print_help()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(dry_run(args.dry_run) if args.dry_run else batch(args.n))


if __name__ == "__main__":
    main()
