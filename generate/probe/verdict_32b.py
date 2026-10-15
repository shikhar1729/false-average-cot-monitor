"""The 32B surrogate monitor's own verdict on the probe population.

Writes data/probe/verdict_32b.jsonl (Section 6: the 32B surrogate's verdict AUC; prepare_inputs_32b.py and
analyze_32b.py also read it). Qwen3-32B on OpenRouter, thinking off (/no_think, as in the probe run), scores
the 150 obfuscated hacks and 150 legit solves of runs/probe/internals_inputs_obf.jsonl. Resumable.
Run: python -m generate.probe.verdict_32b. Needs OPENROUTER_API_KEY. Rough cost: under $1.
"""
import asyncio
import json
import os

import httpx

from generate.common import tw
from generate.common.openrouter import chat
from generate.common.paths import DATA, ROOT

MODEL = "qwen/qwen3-32b"
IN = ROOT / "runs" / "probe" / "internals_inputs_obf.jsonl"
OUT = DATA / "probe" / "verdict_32b.jsonl"


async def verdict(client, sem, row):
    reply = await chat(client, sem, MODEL, row["system"] + "\n\n/no_think", row["user"], max_tokens=1200)
    m = tw.SCORE_RE.search(reply or "")   # the score regex alone, as in the run (not the authors' parser)
    return {"key": f"{row['sample_id']}|{row['variant']}", "variant": row["variant"], "is_hack": row["is_hack"],
            "score": float(m.group(1)) if m else None}


async def run():
    rows = [r for r in (json.loads(line) for line in open(IN) if line.strip()) if r["variant"] in ("obf", "legit")]
    done = {json.loads(line)["key"] for line in open(OUT) if line.strip()} if OUT.exists() else set()
    todo = [r for r in rows if f"{r['sample_id']}|{r['variant']}" not in done]
    print(f"scoring {len(todo)} rows (obf and legit) with {MODEL}", flush=True)
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        with open(OUT, "a") as f:
            tasks = [asyncio.create_task(verdict(client, sem, r)) for r in todo]
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                f.write(json.dumps(await fut) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    assert "OPENROUTER_API_KEY" in os.environ, "set OPENROUTER_API_KEY"
    asyncio.run(run())
