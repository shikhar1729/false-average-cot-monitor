"""Resumable JSONL output: skip rows already written, append new rows as they finish."""
import asyncio
import json


def done(path, key="sid"):
    return {json.loads(line)[key] for line in open(path)} if path.exists() else set()


async def append(path, jobs):
    """Run the jobs concurrently and append each row to path as it finishes."""
    with open(path, "a") as f:
        for i, job in enumerate(asyncio.as_completed([asyncio.create_task(j) for j in jobs]), 1):
            f.write(json.dumps(await job) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"  {path.name}: {i}/{len(jobs)}", flush=True)
