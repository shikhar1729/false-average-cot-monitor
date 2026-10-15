"""Build the probe population: obfuscated and original reasoning-dependent hacks, plus legit solves.

Writes runs/probe/internals_inputs_obf.jsonl, not data/, since it holds obfuscated reasoning. For 150
reasoning-dependent hacks it renders the original trajectory and one fresh gpt-5.4 obfuscation as the
monitor reads them (dataset rubric, full scope), plus 150 legit solves drawn with seed 1. pod_8b.py,
verdict_32b.py, prepare_inputs_32b.py, sonnet_verdict_rescore.py and generate/distill read it. Resumable.
Run: python -m generate.probe.prepare_inputs [--n 150]. Needs OPENROUTER_API_KEY and the TW clone.
Rough cost: about $6 (150 obfuscator calls).
"""
import argparse
import asyncio
import collections
import json
import os

import httpx
import numpy as np

from generate.common import obfuscator, subset, tw
from generate.common.paths import ROOT

OUT = ROOT / "runs" / "probe" / "internals_inputs_obf.jsonl"


def legit_solves(n):
    """A seed-1 draw of n legit solves (not the seed-0 calibration draw of subset.legit)."""
    pool = [s for s in tw.collect() if not s["is_hack"] and s.get("classification") == "legitimate"]
    return [pool[i] for i in np.random.default_rng(1).permutation(len(pool))[:n]]


def row(sample, variant, atif):
    return {"sample_id": tw.sid(sample), "variant": variant, "is_hack": int(variant != "legit"),
            "task_id": sample["task_id"], "system": tw.DEV_MSG, "user": tw.render(atif, sample["task_id"])}


async def hack_rows(client, sem, sample, done):
    """The original render, and an obfuscated render when the rewrite passes the rail."""
    out = []
    if (tw.sid(sample), "orig") not in done:
        out.append(row(sample, "orig", tw.load(sample)))
    if (tw.sid(sample), "obf") not in done:
        obf, _ = await obfuscator.obfuscate(client, sem, sample)
        if obf:
            out.append(row(sample, "obf", obf))
    return out


async def run(n):
    hacks = subset.reasoning_dependent(cap=n)
    legits = legit_solves(n)
    done = set()
    if OUT.exists():
        done = {(r["sample_id"], r["variant"]) for r in map(json.loads, open(OUT))}
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        per_hack = await asyncio.gather(*[hack_rows(client, sem, s, done) for s in hacks])
    rows = [r for rs in per_hack for r in rs]
    for s in legits:
        atif = tw.load(s)
        if (tw.sid(s), "legit") not in done and atif:
            rows.append(row(s, "legit", atif))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = collections.Counter(json.loads(line)["variant"] for line in open(OUT))
    print(f"wrote {OUT}: {sum(counts.values())} rows", dict(counts), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150, help="hacks, and legit solves")
    args = ap.parse_args()
    assert "OPENROUTER_API_KEY" in os.environ, "set OPENROUTER_API_KEY"
    asyncio.run(run(args.n))


if __name__ == "__main__":
    main()
