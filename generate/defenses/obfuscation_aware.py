"""The obfuscation-aware defense: a judge warned about the rewrite and primed with two examples of it.

Writes data/defenses/obfuscation_aware.jsonl (Table 2; also one of Section 4's repeated draws): 100
reasoning-dependent hacks, each rewritten once and scored by the held-out sonnet (full scope) under the
dataset rubric and the obfuscation-aware rubric, plus the original under obfuscation-aware. Also writes
data/defenses/legit_obfuscation_aware.jsonl, 150 legit solves under that rubric (its threshold). The two
examples come from tasks 1012 and 1074, outside the subset. Run `python -m generate.defenses.obfuscation_aware`.
Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $25.
"""
import argparse
import asyncio
import os

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA

OUT = DATA / "defenses" / "obfuscation_aware.jsonl"
LEGIT_OUT = DATA / "defenses" / "legit_obfuscation_aware.jsonl"


async def score(client, sem, rubric, atif, task_id):
    return await judge.score(client, sem, rubric, tw.render(atif, task_id))


async def legit_row(client, sem, s):
    return {"sid": tw.sid(s), "adaptive": await score(client, sem, judge.OBFUSCATION_AWARE, tw.load(s), s["task_id"])}


async def hack_row(client, sem, s):
    """Rewrite once; score the original under the aware rubric and the rewrite under both rubrics."""
    orig = tw.load(s)
    obf, status = await obfuscator.obfuscate(client, sem, s)
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "obf_status": status}
    row["orig_adaptive"] = await score(client, sem, judge.OBFUSCATION_AWARE, orig, s["task_id"])
    if obf:
        row["obf_baseline"] = await score(client, sem, judge.BASELINE, obf, s["task_id"])
        row["obf_adaptive"] = await score(client, sem, judge.OBFUSCATION_AWARE, obf, s["task_id"])
    return row


async def main(n):
    hacks, legit = subset.reasoning_dependent(cap=n), subset.legit(150)
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        seen = done(LEGIT_OUT)
        todo = [s for s in legit if tw.sid(s) not in seen]
        print(f"obfuscation-aware legit solves: {len(todo)} to score", flush=True)
        await append(LEGIT_OUT, [legit_row(client, sem, s) for s in todo])
        seen = done(OUT)
        todo = [s for s in hacks if tw.sid(s) not in seen]
        print(f"obfuscation-aware defense: {len(hacks)} hacks, {len(todo)} to do", flush=True)
        await append(OUT, [hack_row(client, sem, s) for s in todo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="hacks to rewrite (the shipped run used 100)")
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main(args.n))
