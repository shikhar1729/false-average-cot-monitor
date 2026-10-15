"""The intent-blind defense: a full-scope judge told to ignore the agent's stated intent.

Writes data/defenses/intent_blind.jsonl (Table 2; also one of Section 4's repeated draws): all 251
reasoning-dependent hacks, each rewritten once and scored by the held-out sonnet (full scope) under the
dataset, skeptic, consistency and intent-blind rubrics, plus the original under intent-blind. Also
writes data/defenses/legit_intent_blind.jsonl, 150 legit solves under intent-blind (its threshold).
Run `python -m generate.defenses.intent_blind`. Needs OPENROUTER_API_KEY and the Terminal Wrench clone;
roughly $60.
"""
import argparse
import asyncio
import os

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA

OUT = DATA / "defenses" / "intent_blind.jsonl"
LEGIT_OUT = DATA / "defenses" / "legit_intent_blind.jsonl"
RUBRICS = {"baseline": judge.BASELINE, "skeptic": judge.SKEPTIC, "consistency": judge.CONSISTENCY,
           "intentblind": judge.INTENT_BLIND}


async def score(client, sem, rubric, atif, task_id):
    return await judge.score(client, sem, rubric, tw.render(atif, task_id))


async def legit_row(client, sem, s):
    return {"sid": tw.sid(s), "intentblind": await score(client, sem, judge.INTENT_BLIND, tw.load(s), s["task_id"])}


async def hack_row(client, sem, s):
    """Rewrite once; score the original under intent-blind and the rewrite under all four rubrics."""
    orig = tw.load(s)
    obf, status = await obfuscator.obfuscate(client, sem, s)
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "obf_status": status}
    row["orig_intentblind"] = await score(client, sem, judge.INTENT_BLIND, orig, s["task_id"])
    if obf:
        for name, rubric in RUBRICS.items():
            row[f"obf_{name}"] = await score(client, sem, rubric, obf, s["task_id"])
    return row


async def main(n, n_legit):
    hacks, legit = subset.reasoning_dependent(cap=n), subset.legit(n_legit)
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        seen = done(LEGIT_OUT)
        todo = [s for s in legit if tw.sid(s) not in seen]
        print(f"intent-blind legit solves: {len(legit)}, {len(todo)} to score", flush=True)
        await append(LEGIT_OUT, [legit_row(client, sem, s) for s in todo])
        seen = done(OUT)
        todo = [s for s in hacks if tw.sid(s) not in seen]
        print(f"intent-blind defense: {len(hacks)} hacks, {len(todo)} to do", flush=True)
        await append(OUT, [hack_row(client, sem, s) for s in todo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=251)
    ap.add_argument("--neg-cap", type=int, default=150, help="legit solves scored for the threshold")
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main(args.n, args.neg_cap))
