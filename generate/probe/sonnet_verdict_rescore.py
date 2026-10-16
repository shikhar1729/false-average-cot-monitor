"""The held-out sonnet monitor's own verdict on the population the probe reads, three times over.

Writes data/probe/sonnet_verdict_rescore.jsonl (Section 6 and Appendix I: the real monitor's verdict AUC,
0.62). Sonnet (dataset rubric, temperature 0) scores the 100 obfuscated hacks of the nested draw, as rendered
in runs/probe/internals_inputs_obf.jsonl, and the 150 legit calibration solves, 3 repeats each. Resumable.
Run: python -m generate.probe.sonnet_verdict_rescore [--repeats 3]. Needs OPENROUTER_API_KEY and the TW clone.
Rough cost: about $15.
"""
import argparse
import asyncio
import json
import os

import httpx

from generate.common import judge, subset, tw
from generate.common.paths import DATA, ROOT

INPUTS = ROOT / "runs" / "probe" / "internals_inputs_obf.jsonl"
OUT = DATA / "probe" / "sonnet_verdict_rescore.jsonl"


def items():
    """(kind, sid, monitor input): the seed-0 nested draw of 100 obfuscated hacks, then the 150 legits."""
    rows = [json.loads(line) for line in open(INPUTS) if line.strip()]
    assert rows[0]["system"] == tw.DEV_MSG
    obf = {r["sample_id"]: r["user"] for r in rows if r["variant"] == "obf"}
    hacks = subset.reasoning_dependent(cap=100)
    legits = subset.legit(150)
    return ([("obf", tw.sid(s), obf[tw.sid(s)]) for s in hacks]
            + [("legit", tw.sid(s), tw.render(tw.load(s), s["task_id"])) for s in legits])


async def one(client, sem, kind, sid, user, repeat):
    score = await judge.score(client, sem, judge.BASELINE, user, max_tokens=4000)
    return {"kind": kind, "sid": sid, "repeat": repeat, "score": score}


async def run(repeats):
    done = set()
    if OUT.exists():
        done = {(r["kind"], r["sid"], r["repeat"]) for r in (json.loads(line) for line in open(OUT) if line.strip())}
    fixed_set = items()
    todo = [(k, s, u, rep) for rep in range(repeats) for k, s, u in fixed_set if (k, s, rep) not in done]
    print(f"{len(todo)} monitor calls to do", flush=True)
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient() as client:
        with open(OUT, "a") as f:
            tasks = [asyncio.create_task(one(client, sem, *t)) for t in todo]
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                f.write(json.dumps(await fut) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    assert "OPENROUTER_API_KEY" in os.environ, "set OPENROUTER_API_KEY"
    asyncio.run(run(args.repeats))


if __name__ == "__main__":
    main()
