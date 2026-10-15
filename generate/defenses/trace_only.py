"""Trace-only defenses against the rewrite: skeptic, consistency, and oracle-localized consistency.

Writes data/defenses/trace_only.jsonl (Table 2): 150 reasoning-dependent hacks, each rewritten once and
scored by the held-out sonnet (full scope) under the dataset, skeptic and consistency rubrics, and under
consistency with a marker before the dataset's own key hack step (oracle). Also writes the 150 legit
solves behind two thresholds: legit_consistency.jsonl and legit_oracle.jsonl (marker at a seeded random
step); the skeptic threshold comes from generate.attack.headline. Run `python -m generate.defenses.trace_only`.
Needs OPENROUTER_API_KEY and the Terminal Wrench clone; roughly $40.
"""
import argparse
import asyncio
import json
import os
import random

import httpx

from generate.common import judge, obfuscator, subset, tw
from generate.common.jsonl import append, done
from generate.common.paths import DATA, TW

OUT = DATA / "defenses" / "trace_only.jsonl"
LEGIT_CONSISTENCY = DATA / "defenses" / "legit_consistency.jsonl"
LEGIT_ORACLE = DATA / "defenses" / "legit_oracle.jsonl"
RUBRICS = {"baseline": judge.BASELINE, "skeptic": judge.SKEPTIC, "consistency": judge.CONSISTENCY}


def key_hack_steps():
    """sid -> the dataset's key_hack_steps annotation, for every Gemini hack."""
    return {f"{e['task_id']}__{e.get('trajectory_label', 'v5')}__hack": e.get("key_hack_steps")
            for e in json.load(open(TW / "index" / "trajectories.json")) if e["model"] == "gemini-3.1-pro"}


def agent_step_ids(atif):
    return [st.get("step_id") for st in atif.get("steps", []) if st.get("source") == "agent"]


def render_oracle(atif, task_id, targets):
    """Full-scope render with the focus marker inserted before the first target agent step."""
    ids = agent_step_ids(atif)
    if not ids:
        return None
    first = min([k for k in (targets or []) if k in ids] or [ids[0]])
    steps = []
    for st in atif.get("steps", []):
        if st.get("source") == "agent" and st.get("step_id") == first:
            steps.append({"source": "agent", "step_id": "FOCUS", "message": judge.FOCUS_MARKER,
                          "tool_calls": [], "observation": {}})
        steps.append(st)
    return tw.render({**atif, "steps": steps}, task_id)


async def legit_consistency(client, sem, s):
    user = tw.render(tw.load(s), s["task_id"])
    return {"sid": tw.sid(s), "consistency": await judge.score(client, sem, judge.CONSISTENCY, user)}


async def legit_oracle(client, sem, s):
    """The oracle's false-positive control: the marker at an agent step drawn with the sid as seed."""
    atif = tw.load(s)
    ids = agent_step_ids(atif)
    user = render_oracle(atif, s["task_id"], [random.Random(tw.sid(s)).choice(ids)] if ids else None)
    return {"sid": tw.sid(s), "oracle": await judge.score(client, sem, judge.CONSISTENCY, user) if user else None}


async def hack_row(client, sem, s, key_steps):
    orig = tw.load(s)
    obf, status = await obfuscator.obfuscate(client, sem, s)
    row = {"sid": tw.sid(s), "task_id": s["task_id"], "obf_status": status}
    row["orig_consistency"] = await judge.score(client, sem, judge.CONSISTENCY, tw.render(orig, s["task_id"]))
    if obf:
        for name, rubric in RUBRICS.items():
            row[f"obf_{name}"] = await judge.score(client, sem, rubric, tw.render(obf, s["task_id"]))
        user = render_oracle(obf, s["task_id"], key_steps.get(tw.sid(s)))
        row["obf_oracle"] = await judge.score(client, sem, judge.CONSISTENCY, user) if user else None
    return row


async def main(n):
    hacks, legit, key_steps = subset.reasoning_dependent(cap=n), subset.legit(150), key_hack_steps()
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient() as client:
        for path, legit_row in ((LEGIT_CONSISTENCY, legit_consistency), (LEGIT_ORACLE, legit_oracle)):
            seen = done(path)
            await append(path, [legit_row(client, sem, s) for s in legit if tw.sid(s) not in seen])
        seen = done(OUT)
        todo = [s for s in hacks if tw.sid(s) not in seen]
        print(f"trace-only defenses: {len(hacks)} hacks, {len(todo)} to do", flush=True)
        await append(OUT, [hack_row(client, sem, s, key_steps) for s in todo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    asyncio.run(main(args.n))
