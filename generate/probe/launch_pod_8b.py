"""Rent a GPU, run pod_8b.py on runs/probe/internals_inputs_obf.jsonl, fetch the results, always delete the pod.

Default: Qwen3-8B with its verdicts, inputs capped at 16384 tokens; the RESULTS block it fetches into
runs/probe/probe_results_8b.json is data/probe/probe_8b.json (Section 6, Table 6). With --model
Qwen/Qwen3-32B --maxtok 6144 --no-verbal it made runs/probe/probe_feats_32b.pt, the 32B features that
prepare_inputs_32b.py and analyze_32b.py start from.
With --inputs runs/probe/internals_inputs_ao.jsonl --no-verbal --tag ao it made the action-only probe
(see generate/probe/action_only.py).
Run: python -m generate.probe.launch_pod_8b. Needs RUNPOD_API_KEY, HF_TOKEN and runpodctl.
Rough cost: under $1 for 8B (under an hour on an A40); about $3 for 32B on an A100 80GB.
"""
import argparse
import os
import subprocess
from pathlib import Path

from generate.probe import pod

INPUTS = pod.RUNS / "internals_inputs_obf.jsonl"
POD_SCRIPT = Path(__file__).with_name("pod_8b.py")
SMALL_GPUS = ["NVIDIA A40", "NVIDIA RTX A5000", "NVIDIA RTX 4090", "NVIDIA RTX A4000",
              "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
BIG_GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"]


def run_probe(ssh, scp, host, args):
    with open(POD_SCRIPT) as f:
        subprocess.run(ssh + ["cat > /workspace/pod_8b.py"], stdin=f, check=True, timeout=60)
    subprocess.run(scp + [str(args.inputs), f"root@{host}:/workspace/internals_inputs_obf.jsonl"], check=True, timeout=300)
    print("installing deps and running the probe (30 to 50 min) ...", flush=True)
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' accelerate scikit-learn safetensors "
              "huggingface_hub 2>&1 | tail -2 && echo '--- RUN ---' && read -r HFT && HF_TOKEN=$HFT "
              f"PROBE_MODEL={args.model} MAXTOK={args.maxtok} VERBAL={0 if args.no_verbal else 1} python3 pod_8b.py")
    # the HF token goes over stdin, never on a command line
    run = subprocess.run(ssh + [remote], capture_output=True, text=True, timeout=6000,
                         input=os.environ.get("HF_TOKEN", "") + "\n")
    (pod.RUNS / f"pod_8b_{args.tag}_stdout.txt").write_text(run.stdout)
    (pod.RUNS / f"pod_8b_{args.tag}_stderr.txt").write_text(run.stderr)
    print("=== PROBE STDOUT (tail) ===\n" + run.stdout[-3500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-1500:], flush=True)


def fetch(scp, host, tag):
    for remote, local in (("probe_results.json", f"probe_results_{tag}.json"), ("probe_feats.pt", f"probe_feats_{tag}.pt")):
        r = subprocess.run(scp + [f"root@{host}:/workspace/{remote}", str(pod.RUNS / local)],
                           capture_output=True, text=True, timeout=300)
        print(f"scp {remote} -> runs/probe/{local} rc={r.returncode} {r.stderr[-120:]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--maxtok", type=int, default=16384, help="input token cap")
    ap.add_argument("--no-verbal", action="store_true", help="skip the verdict generation (the 32B run)")
    ap.add_argument("--tag", default=None, help="output suffix; default 32b or 8b from --model")
    ap.add_argument("--inputs", type=Path, default=INPUTS, help="probe inputs (generate.probe.action_only writes the action-only set)")
    args = ap.parse_args()
    big = "32B" in args.model
    args.tag = args.tag or ("32b" if big else "8b")
    assert args.inputs.exists(), f"missing {args.inputs}: run generate.probe.prepare_inputs first"
    print(f"model={args.model} tag={args.tag} maxtok={args.maxtok}", flush=True)
    pid, _, _ = pod.create("probe", BIG_GPUS if big else SMALL_GPUS, 160 if big else 60, hours=2)
    try:
        ssh, scp, host = pod.connect(pid)
        run_probe(ssh, scp, host, args)
        fetch(scp, host, args.tag)
    finally:
        pod.delete(pid)


if __name__ == "__main__":
    main()
