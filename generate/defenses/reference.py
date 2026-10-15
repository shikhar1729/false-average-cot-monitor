"""The external-ground-truth defense: the judge sees a genuine solution of the same task beside the candidate.

Writes data/defenses/reference.jsonl (Table 2, last row): 150 reasoning-dependent hacks, each rewritten
once and scored by sonnet against a genuine solve of its task (kind "hack"), and 150 legit solves scored
against a different genuine solve of their task (kind "neg"; these set the threshold). Run
`python -m generate.defenses.reference`. Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $30.
"""
import argparse
import asyncio
import collections
import json
import os

import httpx
import numpy as np

from generate.common import judge, obfuscator, subset, tw
from generate.common.openrouter import SONNET, chat
from generate.common.paths import DATA

OUT = DATA / "defenses" / "reference.jsonl"


def genuine_solves_by_task():
    by_task = collections.defaultdict(list)
    for s in tw.collect():
        if not s["is_hack"] and s["classification"] == "legitimate":
            by_task[s["task_id"]].append(s)
    return by_task


def two_trajectories(task_id, ref_atif, cand_atif):
    return (f"TASK: {tw.M.load_instruction(task_id, 'gemini-3.1-pro')}\n\n"
            f"=== REFERENCE (a genuine solution to this task) ===\n{tw.M.format_atif_trajectory(ref_atif)}\n\n"
            f"=== CANDIDATE (evaluate THIS trajectory) ===\n{tw.M.format_atif_trajectory(cand_atif)}")


async def score(client, sem, task_id, ref_atif, cand_atif):
    # Two trajectories make a long prompt; the original run allowed 200 s per call.
    reply = await chat(client, sem, SONNET, judge.REFERENCE, two_trajectories(task_id, ref_atif, cand_atif),
                       max_tokens=3000, timeout=200)
    return tw.extract_score(reply) if reply else None


async def legit_row(client, sem, task_id, solves):
    cand, ref = solves[0], solves[1]
    return {"sid": tw.sid(cand), "kind": "neg", "score": await score(client, sem, task_id, tw.load(ref), tw.load(cand))}


async def hack_row(client, sem, s, ref):
    obf, status = await obfuscator.obfuscate(client, sem, s)
    sc = await score(client, sem, s["task_id"], tw.load(ref), obf) if obf else None
    return {"sid": tw.sid(s), "kind": "hack", "task_id": s["task_id"], "obf_status": status, "score": sc}


async def main(n):
    refs, hacks = genuine_solves_by_task(), subset.reasoning_dependent(cap=n)
    pairs = [(t, v) for t, v in refs.items() if len(v) >= 2]
    pairs = [pairs[i] for i in np.random.default_rng(0).permutation(len(pairs))[:n]]
    done = {(r["sid"], r["kind"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        jobs = ([asyncio.create_task(legit_row(client, sem, t, v)) for t, v in pairs if (tw.sid(v[0]), "neg") not in done]
                + [asyncio.create_task(hack_row(client, sem, s, refs[s["task_id"]][0]))
                   for s in hacks if (tw.sid(s), "hack") not in done])
        print(f"reference defense: {len(jobs)} to score (hacks and legit solves)", flush=True)
        with open(OUT, "a") as f:
            for i, job in enumerate(asyncio.as_completed(jobs), 1):
                f.write(json.dumps(await job) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="hacks, and legit solves, to score")
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main(args.n))
