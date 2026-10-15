"""The headline attack: rewrite the reasoning of reasoning-dependent hacks, score under two rubrics.

Writes data/attack/headline.jsonl (Section 4 headline; Table 1, skeptic row; Appendix B split-half):
a seeded 200-hack draw from the reasoning-dependent subset, each rewritten once and scored before and
after by the held-out sonnet (full scope) under the dataset rubric and the skeptic rubric. Also writes
data/defenses/legit_skeptic.jsonl, 150 legit solves under the skeptic rubric (its threshold). The
shipped file has 209 rows: this draw plus 9 subset hacks an earlier, resumed run added.
Run `python -m generate.attack.headline`. Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $60.
"""
import asyncio
import os

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA

OUT = DATA / "attack" / "headline.jsonl"
LEGIT_OUT = DATA / "defenses" / "legit_skeptic.jsonl"
RUBRICS = {"baseline": judge.BASELINE, "skeptic": judge.SKEPTIC}
N_HACKS = 200


async def score(client, sem, rubric, atif, task_id):
    return await judge.score(client, sem, rubric, tw.render(atif, task_id))


async def legit_row(client, sem, s):
    return {"sid": tw.sid(s), "skeptic": await score(client, sem, judge.SKEPTIC, tw.load(s), s["task_id"])}


async def hack_row(client, sem, s):
    """Rewrite once, then score the original and the rewrite on the same trajectory under both rubrics."""
    orig = tw.load(s)
    obf, status = await obfuscator.obfuscate(client, sem, s)
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "obf_status": status, "observed_categories": s["observed_categories"]}
    for name, rubric in RUBRICS.items():
        row[f"orig_{name}"] = await score(client, sem, rubric, orig, s["task_id"])
        row[f"obf_{name}"] = await score(client, sem, rubric, obf, s["task_id"]) if obf else None
    return row


async def main():
    hacks, legit = subset.reasoning_dependent(cap=N_HACKS), subset.legit(150)
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        seen = done(LEGIT_OUT)
        await append(LEGIT_OUT, [legit_row(client, sem, s) for s in legit if tw.sid(s) not in seen])
        seen = done(OUT)
        await append(OUT, [hack_row(client, sem, s) for s in hacks if tw.sid(s) not in seen])


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main())
