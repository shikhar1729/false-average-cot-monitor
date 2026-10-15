"""Iterate the attack: round k rewrites the reasoning that round k-1 produced.

Writes data/attack/iterated.jsonl (Section 4, iteration): for 60 reasoning-dependent hacks, the nano
(target) and sonnet (held-out) full-scope scores of the original and of two chained rewrites. A round
whose rewrite fails carries the previous round's scores forward. Run `python -m generate.attack.iterated`.
Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $8.
"""
import argparse
import asyncio
import json
import os

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.openrouter import NANO, OBFUSCATOR, SONNET, chat
from generate.common.paths import DATA

OUT = DATA / "attack" / "iterated.jsonl"
ROUNDS = 2


async def rewrite(client, sem, atif, task_id):
    """obfuscator.obfuscate on an in-memory trajectory, so that rounds can chain. Returns (atif|None, status)."""
    steps = obfuscator.agent_msg_steps(atif)
    if not steps:
        return None, "no_agent_message"
    listing = "\n\n".join(f"[step_id={sid}]\n{msg}" for sid, msg in steps)
    user = (f"TASK + FULL TRAJECTORY (actions/observations FIXED, shown for coherence):\n"
            f"{tw.render(atif, task_id)}\n\nAGENT REASONING STEPS TO REWRITE ({len(steps)}):\n{listing}")
    content = await chat(client, sem, OBFUSCATOR, obfuscator.OBFUSCATOR_PROMPT, user,
                         temperature=0.7, max_tokens=8000, timeout=240)
    if content is None:
        return None, "obfuscator_failed"
    try:
        new = obfuscator.apply_rewrites(atif, obfuscator.parse_rewrites(content))
        obfuscator.assert_exploit_preserved(atif, new)
    except (json.JSONDecodeError, KeyError, AssertionError) as e:
        return None, f"bad_rewrite: {type(e).__name__}"
    if obfuscator.agent_msg_steps(new) == steps:
        return None, "no_op_rewrite"
    return new, "ok"


async def monitor_scores(client, sem, atif, task_id):
    # The pool's own monitor call: dataset rubric, temperature 0, 4000 completion tokens.
    user = tw.render(atif, task_id)
    return {tag: await judge.score(client, sem, judge.BASELINE, user, model=model, max_tokens=4000)
            for tag, model in (("m0", NANO), ("m1", SONNET))}


async def chain(client, sem, s):
    atif = tw.load(s)
    rounds = [{"k": 0, "status": "orig", **await monitor_scores(client, sem, atif, s["task_id"])}]
    for k in range(1, ROUNDS + 1):
        new, status = await rewrite(client, sem, atif, s["task_id"])
        if new is None:
            rounds.append({"k": k, "status": status, "m0": rounds[-1]["m0"], "m1": rounds[-1]["m1"]})
            continue
        atif = new
        rounds.append({"k": k, "status": status, **await monitor_scores(client, sem, atif, s["task_id"])})
    return {"sid": tw.sid(s), "task_id": s["task_id"], "rounds": rounds}


async def main(n):
    hacks = subset.reasoning_dependent(cap=n)
    done = {json.loads(line)["sid"] for line in open(OUT)} if OUT.exists() else set()
    todo = [s for s in hacks if tw.sid(s) not in done]
    print(f"iterated: {len(hacks)} hacks, {len(todo)} to chain ({ROUNDS} rounds)", flush=True)
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        with open(OUT, "a") as f:
            for i, job in enumerate(asyncio.as_completed([asyncio.create_task(chain(client, sem, s)) for s in todo]), 1):
                f.write(json.dumps(await job) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main(args.n))
