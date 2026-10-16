"""The other-actor replication under a monitor that does not saturate on Opus's honest solves.

Writes data/attack/opus_gpt54.jsonl (Table 1, gpt-5.4 row; Appendix B): the 97 opus hacks of
data/attack/opus_actor.jsonl, given a fresh obfuscated and paraphrased rewrite and scored by gpt-5.4
(full scope, dataset rubric; its threshold comes from data/opus_pool/gpt54_full_honest.jsonl), with
the originals re-scored the same day. Both rewrites are also scored by flash, as a check that Table 1's
flash row reproduces. Run `python -m generate.attack.opus_gpt54`. Needs OPENROUTER_API_KEY and the
Terminal Wrench clone; roughly $15.
"""
import asyncio
import json
import os

import httpx

from generate.common import judge, obfuscator, tw
from generate.common.jsonl import append, done
from generate.common.openrouter import FLASH
from generate.common.paths import DATA

ACTOR = "claude-opus-4.6"
GPT54 = "openai/gpt-5.4"
OUT = DATA / "attack" / "opus_gpt54.jsonl"
VARIANTS = {"obf": obfuscator.OBFUSCATOR_PROMPT, "par": obfuscator.PARAPHRASE_PROMPT}


async def hack_row(client, sem, s):
    async def score(atif, model):
        return await judge.score(client, sem, judge.BASELINE, tw.render(atif, s["task_id"], ACTOR), model=model, max_tokens=4000)
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "orig": await score(tw.load(s), GPT54)}
    for variant, system in VARIANTS.items():
        new, row[f"{variant}_status"] = await obfuscator.obfuscate(client, sem, s, system, actor=ACTOR)
        row[variant] = await score(new, GPT54) if new else None
        row[f"{variant}_flash"] = await score(new, FLASH) if new else None
    return row


async def main():
    want = {json.loads(line)["sid"] for line in open(DATA / "attack" / "opus_actor.jsonl")}
    hacks = [s for s in tw.collect(ACTOR) if s["is_hack"] and tw.sid(s) in want]
    seen, sem = done(OUT), asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        await append(OUT, [hack_row(client, sem, s) for s in hacks if tw.sid(s) not in seen])


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main())
