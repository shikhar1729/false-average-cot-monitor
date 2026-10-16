"""The dataset's own sanitizer as a baseline for the rewrite.

Writes data/attack/sanitized.jsonl (Table 1, sanitizer row; Appendix B): for the 209 headline hacks,
the held-out sonnet (full scope, dataset rubric) scores the original and the sanitized copy Terminal
Wrench ships (sanitized_trajectories/, made by its monitoring/sanitize.py: one LLM call per agent
message, removing references to hacking). Commands and outputs are identical in the two copies.
Run `python -m generate.attack.sanitized`. Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $7.
"""
import asyncio
import json
import os

import httpx

from generate.common import judge, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA, TW

OUT = DATA / "attack" / "sanitized.jsonl"
ACTOR = "gemini-3.1-pro"


def sanitized_path(s):
    return TW / "tasks" / s["task_id"] / ACTOR / "sanitized_trajectories" / s["label"] / "trial" / "agent" / "trajectory.json"


async def hack_row(client, sem, s):
    async def score(atif):
        return await judge.score(client, sem, judge.BASELINE, tw.render(atif, s["task_id"]))
    sanitized = tw.load({"path": str(sanitized_path(s))})
    return {"sid": tw.sid(s), "task_id": s["task_id"], "orig": await score(tw.load(s)), "san": await score(sanitized)}


async def main():
    want = {json.loads(line)["sid"] for line in open(DATA / "attack" / "headline.jsonl")}
    hacks = [s for s in tw.collect(ACTOR) if s["is_hack"] and tw.sid(s) in want]
    assert all(sanitized_path(s).exists() for s in hacks), "a headline hack has no sanitized copy"
    seen, sem = done(OUT), asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        await append(OUT, [hack_row(client, sem, s) for s in hacks if tw.sid(s) not in seen])


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main())
