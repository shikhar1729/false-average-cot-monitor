"""Run one live agent over every task that passed the sandbox sweep (Section 5, Table 5): shard the 27
tasks round-robin over k self-cleaning pods, each running generate.live.agents, wait, then pool the
shard files (runs/live/<agent>_s<i>.json) into data/live/<agent>.json. Resumable: rerun with the same
arguments and every shard resumes where it stopped.
  python -m generate.live.launch --agent anthropic/claude-opus-4.6 [--k 5] [--nb 6 --na 2 --nh 2]
  python -m generate.live.launch --agent anthropic/claude-opus-4.6 --pool-only   # $0: pool existing shards
Needs the TW clone, OPENROUTER_API_KEY and RUNPOD_API_KEY. API cost was $28 to $89 per agent ($173 for all three), plus pod time.
"""
import argparse
import json
import subprocess
import sys
import time

from generate.common.paths import DATA, ROOT

RUNS = ROOT / "runs" / "live"
SWEEP = DATA / "live" / "task_sweep.json"


def launch(args, groups, slug):
    procs = []
    for i, g in enumerate(groups):
        suf = f"_s{i}"
        out = open(RUNS / f"{slug}{suf}.stdout", "a")
        p = subprocess.Popen([sys.executable, "-m", "generate.live.agents", "--agent", args.agent, "--tasks", ",".join(g),
                              "--nb", str(args.nb), "--na", str(args.na), "--nh", str(args.nh), "--suffix", suf],
                             cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        procs.append((i, g, p))
        print(f"  launched shard s{i} pid={p.pid} ({len(g)} tasks): {g}", flush=True)
        time.sleep(15)   # stagger pod creates
    for i, g, p in procs:
        p.wait()
        print(f"  shard s{i} finished rc={p.returncode}", flush=True)


def pool(k, slug):
    recs = []
    for i in range(k):
        f = RUNS / f"{slug}_s{i}.json"
        if f.exists():
            recs += json.load(open(f))
        else:
            print(f"  MISSING shard file {f.name}", flush=True)
    if not recs:
        sys.exit("no shard files to pool")
    out = DATA / "live" / f"{slug}.json"
    out.write_text(json.dumps(recs, separators=(",", ":")))
    print(f"pooled {len(recs)} records -> {out.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--k", type=int, default=5, help="pods (shards)")
    ap.add_argument("--nb", type=int, default=6, help="B_innocent rollouts per task")
    ap.add_argument("--na", type=int, default=2, help="A_control rollouts per task")
    ap.add_argument("--nh", type=int, default=2, help="H_honest rollouts per task")
    ap.add_argument("--pool-only", action="store_true", help="$0: pool existing shard files; launch nothing")
    args = ap.parse_args()
    slug = args.agent.split("/")[-1]
    passing = [r["task"] for r in json.load(open(SWEEP)) if r.get("ok")]
    groups = [passing[i::args.k] for i in range(args.k)]          # round-robin split for balance
    print(f"{args.agent}: {len(passing)} tasks over {args.k} pods, nb={args.nb} na={args.na} nh={args.nh} -> "
          f"{len(passing)*(args.nb+args.na+args.nh)} rollouts", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if not args.pool_only:
        launch(args, groups, slug)
    pool(args.k, slug)


if __name__ == "__main__":
    main()
