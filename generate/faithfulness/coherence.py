"""Does the rewritten reasoning stay faithful to the unchanged actions?

Writes data/faithfulness/coherence.jsonl (Section 5 and Appendix E, "the rewrite stays action-faithful").
For each of 40 reasoning-dependent hacks it obfuscates the reasoning, scores the rewrite with the
held-out sonnet monitor (full scope), and has an independent gpt-5.4 judge rate how faithfully the
rewritten and the original reasoning describe the commands and outputs. Failed rewrites are dropped.
Run: python -m generate.faithfulness.coherence [--n 40]. Needs OPENROUTER_API_KEY and the TW clone.
Rough cost: about $3 (40 rewrites, 40 monitor calls, 80 judge calls).
"""
import argparse
import asyncio
import json
import os

import httpx

from generate.common import obfuscator, subset, tw
from generate.common.paths import DATA
from generate.faithfulness.prompts import held_out_monitor, rate

FAITH_JUDGE = "openai/gpt-5.4"
OUT = DATA / "faithfulness" / "coherence.jsonl"


async def one(client, sem, sample):
    """The rewrite's monitor score and both faithfulness ratings, or None if the rewrite failed."""
    orig = tw.load(sample)
    obf, _ = await obfuscator.obfuscate(client, sem, sample)
    if obf is None:
        return None
    monitor = await held_out_monitor(client, sem, obf, sample["task_id"])
    faith_obf = await rate(client, sem, FAITH_JUDGE, tw.M.format_atif_trajectory(obf))
    faith_orig = await rate(client, sem, FAITH_JUDGE, tw.M.format_atif_trajectory(orig))
    return {"sid": tw.sid(sample), "monitor_obf": monitor, "faith_obf": faith_obf, "faith_orig": faith_orig}


async def run(n):
    hacks = subset.reasoning_dependent(cap=n)
    print(f"{len(hacks)} reasoning-dependent hacks", flush=True)
    sem = asyncio.Semaphore(6)
    rows = []
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(one(client, sem, s)) for s in hacks]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            row = await fut
            if row:
                rows.append(row)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40, help="number of reasoning-dependent hacks")
    args = ap.parse_args()
    assert "OPENROUTER_API_KEY" in os.environ, "set OPENROUTER_API_KEY"
    asyncio.run(run(args.n))


if __name__ == "__main__":
    main()
