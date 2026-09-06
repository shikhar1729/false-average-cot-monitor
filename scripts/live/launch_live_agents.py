"""Live-bench orchestrator: shard the 27 env-sanity-passing tasks over K self-cleaning live_agents
pods (one agent model per launcher run), wait, pool the shard files into results/scores/live_<slug>_pooled.json
and print the per-condition analysis. Resume-capable: rerun with the same args and each shard resumes.

  python scripts/live/launch_live_agents.py --agent anthropic/claude-opus-4.6 [--k 5] [--nb 6 --na 2 --nh 2]
  python scripts/live/launch_live_agents.py --agent anthropic/claude-opus-4.6 --pool-only        # $0: pool + analyze existing shards
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
RUNS = ROOT / "runs"; RUNS.mkdir(exist_ok=True)   # logs, pod ids, shard stdout (gitignored)
SCORES = ROOT / "results/scores"
ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--nb", type=int, default=6); ap.add_argument("--na", type=int, default=2); ap.add_argument("--nh", type=int, default=2)
ap.add_argument("--pool-only", action="store_true")
A = ap.parse_args()
SLUG = A.agent.split("/")[-1]

passing = [r["task"] for r in json.load(open(SCORES / "env_sanity_sweep.json")) if r.get("ok")]
groups = [passing[i::A.k] for i in range(A.k)]          # round-robin split for balance
print(f"{A.agent}: {len(passing)} tasks over {A.k} pods, nb={A.nb} na={A.na} nh={A.nh} -> "
      f"{len(passing)*(A.nb+A.na+A.nh)} rollouts", flush=True)

if not A.pool_only:
    procs = []
    for i, g in enumerate(groups):
        suf = f"_s{i}"
        out = open(RUNS / f"live_{SLUG}{suf}.stdout", "a")
        p = subprocess.Popen([sys.executable, str(HERE / "live_agents.py"), "--agent", A.agent, "--tasks", ",".join(g),
                              "--nb", str(A.nb), "--na", str(A.na), "--nh", str(A.nh), "--suffix", suf],
                             stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        procs.append((i, g, p))
        print(f"  launched shard s{i} pid={p.pid} ({len(g)} tasks): {g}", flush=True)
        time.sleep(15)   # stagger pod creates
    for i, g, p in procs:
        p.wait()
        print(f"  shard s{i} finished rc={p.returncode}", flush=True)

# ---- pool + analyze (analyze(), imported behind a benign argv; its import opens live_<slug>_pooled.log) ----
recs = []
for i in range(A.k):
    f = SCORES / f"live_{SLUG}_s{i}.json"
    if f.exists():
        recs += json.load(open(f))
    else:
        print(f"  MISSING shard file {f.name}", flush=True)
pooled = SCORES / f"live_{SLUG}_pooled.json"
if recs:
    json.dump(recs, open(pooled, "w"), indent=2)
    print(f"pooled {len(recs)} records -> {pooled.name}", flush=True)
elif pooled.exists():
    recs = json.load(open(pooled))
    print(f"no shard files; analyzing the existing {pooled.name} ({len(recs)} records)", flush=True)
else:
    sys.exit("no shard files and no pooled file to analyze")
sys.argv = [sys.argv[0], "--agent", A.agent, "--suffix", "_pooled"]
import live_agents as LA  # noqa: E402
LA.analyze(recs)
