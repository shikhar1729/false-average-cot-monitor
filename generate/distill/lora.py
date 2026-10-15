"""Score the distillation test with the held-out sonnet monitor. Writes data/distill/lora.jsonl (Section 7).

Each of the 29 held-out trajectories is rebuilt four times with the same commands and outputs and a
different reasoning source per step: the honest original, the obfuscated target, the un-tuned base
model's narration and the LoRA model's narration (runs/distill/sft_gen.json from launch_pod). The monitor
scores each version; one row per {sample_id, source, score}.
Run: python -m generate.distill.lora. Needs OPENROUTER_API_KEY. Rough cost: 116 sonnet calls, about $2.
"""
import asyncio
import json
import os

import httpx

from generate.common import judge, tw
from generate.common.openrouter import SONNET, chat
from generate.common.paths import DATA, ROOT

SOURCES = ("honest", "obf", "base_gen", "sft_gen")
GENERATIONS = ROOT / "runs" / "distill" / "sft_gen.json"
OUT = DATA / "distill" / "lora.jsonl"


def by_trajectory(rows):
    """Held-out steps grouped by sample_id (first-seen order), each group in step order."""
    traj = {}
    for r in rows:
        traj.setdefault(r["sample_id"], []).append(r)
    for steps in traj.values():
        steps.sort(key=lambda r: r["step"])
    return traj


def rebuild(steps, source):
    """The trajectory text with every step's reasoning taken from one source."""
    def reasoning(s):
        return s[source] if isinstance(s.get(source), str) and s[source].strip() else "(no reasoning provided)"
    return "\n\n".join(f"[AGENT]\n{reasoning(s)}\n[ACTIONS]\n{s['cmds']}\n[OBSERVATION]\n{s['output']}" for s in steps)


async def score(client, sem, task, trajectory):
    user = tw.M.USER_TEMPLATE.format(task_description=task, trajectory=trajectory)
    reply = await chat(client, sem, SONNET, judge.BASELINE, user, max_tokens=1500, timeout=120)
    return tw.extract_score(reply) if reply else None


async def run():
    rows = json.load(open(GENERATIONS))
    traj = by_trajectory(rows)
    print(f"{len(rows)} held-out steps over {len(traj)} trajectories", flush=True)
    jobs = [(sid, source, steps[0]["task"], rebuild(steps, source)) for sid, steps in traj.items() for source in SOURCES]
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        scores = await asyncio.gather(*(score(client, sem, task, text) for _, _, task, text in jobs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for (sid, source, _, _), s in zip(jobs, scores):
            f.write(json.dumps({"sample_id": sid, "source": source, "score": s}) + "\n")
    print(f"wrote {len(jobs)} rows to {OUT}", flush=True)


def main():
    assert "OPENROUTER_API_KEY" in os.environ, "set OPENROUTER_API_KEY"
    asyncio.run(run())


if __name__ == "__main__":
    main()
