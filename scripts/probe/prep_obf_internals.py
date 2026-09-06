"""Exp #1 Stage 1 (local, paid): build the white-box-probe input set. For each action-invisible
hack, obfuscate the CoT (gpt-5.4) and render BOTH the obfuscated and original trajectory as the
monitor sees them (system=v1 prompt, user=full render); plus legit-solve negatives. Ship this to a
GPU pod (Stage 2) to extract an open monitor's activations and probe hack-vs-legit.

Question the probe answers: does an open monitor's ACTIVATIONS separate obfuscated-hacks from legits
even though its VERBAL output is fooled (~4% catch)?

Output: results/internals_inputs_obf.jsonl  {sample_id, variant(obf|orig|legit), is_hack, task_id, system, user}
  python scripts/probe/prep_obf_internals.py --n 150
"""
import argparse, asyncio, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
import run_monitor as R
import monitor as M
import obfuscate as O
from skeptic_test import action_invisible_subset
import httpx

OUT = ROOT / "results/internals_inputs_obf.jsonl"
SYSTEM = R.DEV_MSG  # authors' v1 monitor prompt (same the black-box judge uses)


async def run(n):
    assert "OPENROUTER_API_KEY" in os.environ
    R.ACTOR = O.ACTOR
    hacks = action_invisible_subset(cap=n)
    legits = [s for s in R.collect() if not s["is_hack"] and s.get("classification") == "legitimate"]
    legits = [legits[i] for i in np.random.default_rng(1).permutation(len(legits))[:n]]
    done = {(json.loads(l)["sample_id"], json.loads(l)["variant"]) for l in open(OUT)} if OUT.exists() else set()
    sem = asyncio.Semaphore(8)
    rows = []
    async with httpx.AsyncClient() as client:
        async def do_hack(s):
            out = []
            orig = M.load_atif_trajectory(Path(s["path"]))
            if (R.sid(s), "orig") not in done:
                out.append({"sample_id": R.sid(s), "variant": "orig", "is_hack": 1, "task_id": s["task_id"],
                            "system": SYSTEM, "user": O.render_atif(orig, s["task_id"])})
            if (R.sid(s), "obf") not in done:
                obf, status = await O.obfuscate(s, sem, client)
                if obf:
                    out.append({"sample_id": R.sid(s), "variant": "obf", "is_hack": 1, "task_id": s["task_id"],
                                "system": SYSTEM, "user": O.render_atif(obf, s["task_id"])})
            return out
        results = await asyncio.gather(*[do_hack(s) for s in hacks])
        for r in results:
            rows.extend(r)
    for s in legits:
        if (R.sid(s), "legit") in done:
            continue
        atif = M.load_atif_trajectory(Path(s["path"]))
        if atif:
            rows.append({"sample_id": R.sid(s), "variant": "legit", "is_hack": 0, "task_id": s["task_id"],
                         "system": SYSTEM, "user": O.render_atif(atif, s["task_id"])})
    with open(OUT, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import collections
    allrows = [json.loads(l) for l in open(OUT)]
    print(f"wrote {OUT}: {len(allrows)} rows", dict(collections.Counter(r["variant"] for r in allrows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150)
    asyncio.run(run(ap.parse_args().n))
