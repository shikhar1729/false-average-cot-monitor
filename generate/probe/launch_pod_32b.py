"""Rent a GPU, run pod_32b.py on one part of the 32B battery, fetch its outputs, always delete the pod.

Reads runs/probe/inputs_32b_<part>.jsonl and dirs_32b.pt (from prepare_inputs_32b.py) and writes
runs/probe/out_32b_<part>.jsonl and feats_32b_<part>.pt for analyze_32b.py (Section 6, Appendix I). The
two parts can run at once. --cost-cap sets the pod's run deadline from the GPU's hourly price; the pod
also carries a --terminate-after backstop of at most 3 hours, set at creation.
Run: python -m generate.probe.launch_pod_32b --part A (then --part B). Needs RUNPOD_API_KEY, HF_TOKEN and
runpodctl. Rough cost: under $3 per part on one A100 80GB.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

from generate.probe import pod

MODEL, CAP2 = "Qwen/Qwen3-32B", 6144     # CAP2: also read features at the first 32B run's token cap
GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
PRICE = {"NVIDIA A100 80GB PCIe": 1.64, "NVIDIA A100-SXM4-80GB": 1.99, "NVIDIA H100 80GB HBM3": 2.99,
         "NVIDIA H100 PCIe": 2.69}      # conservative fallbacks if the API omits costPerHr
POD_SCRIPT = Path(__file__).with_name("pod_32b.py")


def run_battery(ssh, scp, host, part, run_seconds):
    for local, remote in ((pod.RUNS / f"inputs_32b_{part}.jsonl", "inputs_32b.jsonl"),
                          (pod.RUNS / "dirs_32b.pt", "dirs_32b.pt"), (POD_SCRIPT, "pod_32b.py")):
        subprocess.run(scp + [str(local), f"root@{host}:/workspace/{remote}"], check=True, timeout=300)
    # the HF token goes over stdin into a mode-600 file the remote shell sources, never onto a command line
    subprocess.run(ssh + ["umask 077 && cat > /workspace/.hf_env"], input=f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}\n",
                   text=True, check=True, timeout=60)
    print(f"installing deps and running (deadline {run_seconds / 3600:.2f} h on the pod) ...", flush=True)
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' accelerate safetensors huggingface_hub 2>&1 | tail -1 && "
              "echo '--- RUN ---' && set -a && . /workspace/.hf_env && set +a && "
              f"PROBE_MODEL={MODEL} CAP2={CAP2} RUN_SECONDS={run_seconds} python3 pod_32b.py; rc=$?; rm -f /workspace/.hf_env; exit $rc")
    run = subprocess.run(ssh + [remote], capture_output=True, text=True, timeout=int(run_seconds + 1500))
    (pod.RUNS / f"pod_32b_{part}_stdout.txt").write_text(run.stdout)
    (pod.RUNS / f"pod_32b_{part}_stderr.txt").write_text(run.stderr)
    print("=== STDOUT (tail) ===\n" + run.stdout[-2500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-1500:], flush=True)


def fetch(scp, host, part):
    """Whatever the pod wrote, also after a failure or a deadline stop."""
    for remote, local in (("out_32b.jsonl", f"out_32b_{part}.jsonl"), ("feats_32b.pt", f"feats_32b_{part}.pt")):
        r = subprocess.run(scp + [f"root@{host}:/workspace/{remote}", str(pod.RUNS / local)],
                           capture_output=True, text=True, timeout=600)
        print(f"scp {remote} -> runs/probe/{local} rc={r.returncode} {r.stderr[-120:]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", required=True, choices=["A", "B"])
    ap.add_argument("--cost-cap", type=float, default=2.9, help="USD for the run; 0 = run until the backstop")
    ap.add_argument("--backstop-h", type=float, default=3.0)
    args = ap.parse_args()
    assert args.backstop_h <= 3.0
    for f in (f"inputs_32b_{args.part}.jsonl", "dirs_32b.pt"):
        assert (pod.RUNS / f).exists(), f"missing runs/probe/{f}: run generate.probe.prepare_inputs_32b first"
    t0 = time.time()
    pid, gpu, price = pod.create(f"probe32b_{args.part}", GPUS, 160, hours=args.backstop_h)   # bf16 weights ~65 GB
    price = float(price) if price else PRICE[gpu]
    hours_cap = args.cost_cap / price if args.cost_cap else args.backstop_h
    connected = None
    try:
        connected = pod.connect(pid)
        ssh, scp, host = connected
        run_seconds = max(600, int(hours_cap * 3600 - (time.time() - t0) - 480))   # 8 min left to fetch and delete
        run_battery(ssh, scp, host, args.part, run_seconds)
    finally:
        try:
            if connected:
                fetch(connected[1], connected[2], args.part)
        except Exception as e:
            print("fetch failed:", e, flush=True)
        pod.delete(pid)
        print(f"elapsed {(time.time() - t0) / 3600:.2f} h, ~{(time.time() - t0) / 3600 * price:.2f} USD", flush=True)


if __name__ == "__main__":
    main()
