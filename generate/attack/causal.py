"""A causal (online) rewrite, to bound the one-shot rewrite's hindsight.

Writes data/attack/causal.jsonl (Table 1, causal row; Appendix B): a seed-0 draw of 50 of the 209
headline hacks. Each is rewritten twice by the same obfuscator prompt, once in one pass as in Section 4
and once step by step, where a step's rewrite never sees an output its step has not yet produced.
The held-out sonnet (full scope, dataset rubric) scores the original and both rewrites the same day.
Run `python -m generate.attack.causal`. Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $5.
"""
import asyncio
import json
import os

import httpx
import numpy as np

from generate.common import judge, obfuscator, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA

OUT = DATA / "attack" / "causal.jsonl"
N_HACKS = 50


def draw(n):
    sids = sorted(json.loads(line)["sid"] for line in open(DATA / "attack" / "headline.jsonl"))
    keep = {sids[i] for i in np.random.default_rng(0).permutation(len(sids))[:n]}
    return sorted((s for s in tw.collect() if s["is_hack"] and tw.sid(s) in keep), key=tw.sid)


async def hack_row(client, sem, s):
    async def score(atif):
        return await judge.score(client, sem, judge.BASELINE, tw.render(atif, s["task_id"]))
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "orig": await score(tw.load(s))}
    for name, rewrite in (("oneshot", obfuscator.obfuscate(client, sem, s)), ("causal", obfuscator.obfuscate_causal(client, sem, s))):
        new, row[f"{name}_status"] = await rewrite
        row[name] = await score(new) if new else None
    return row


async def main():
    seen, sem = done(OUT), asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        await append(OUT, [hack_row(client, sem, s) for s in draw(N_HACKS) if tw.sid(s) not in seen])


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main())
