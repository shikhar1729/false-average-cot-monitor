"""Run the LoRA fine-tune and the held-out narration on a rented GPU pod (Section 7).

Creates a RunPod pod with a 2-hour terminate-after backstop, uploads train_on_pod.py and the data from
generate.distill.prepare, runs it, and copies sft_gen.json back to runs/distill/ (with the pod id and
logs). The pod is always deleted at the end, even on failure. generate.distill.lora then scores it.
Run: python -m generate.distill.launch_pod [--model Qwen/Qwen3-8B] [--epochs 3].
Needs runpodctl and RUNPOD_API_KEY (HF_TOKEN optional) in the environment. Rough cost: about one
A100-80GB hour, a few dollars.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from generate.common.paths import ROOT

RUNS = ROOT / "runs" / "distill"
POD_SCRIPT = Path(__file__).with_name("train_on_pod.py")
DATA_FILES = ("sft_train.jsonl", "sft_heldout.jsonl")
GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA H100 80GB HBM3"]
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]


def runpodctl(*args, timeout=180):
    """runpodctl reads RUNPOD_API_KEY from the inherited environment."""
    return subprocess.run(["runpodctl", *args], capture_output=True, text=True, timeout=timeout)


def extract_id(stdout):
    d = json.loads(stdout)
    if isinstance(d, dict):
        return d.get("id") or (d.get("pod") or {}).get("id")
    if isinstance(d, list) and d:
        return d[0]["id"]
    raise KeyError("no id")


def create_pod():
    """The first GPU type with capacity wins; the pod terminates itself after 2 hours regardless."""
    term_after = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for gpu in GPUS:
        print(f"trying {gpu} (auto-terminate {term_after}) ...", flush=True)
        r = runpodctl("pod", "create", "--name", "sft-gen", "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
                      "--container-disk-in-gb", "100", "--terminate-after", term_after, "-o", "json")
        try:
            pod_id = extract_id(r.stdout)
            print(f"created on {gpu} -> {pod_id}", flush=True)
            return pod_id
        except Exception:
            print("  unavailable:", (r.stdout + r.stderr)[-160:].replace("\n", " "), flush=True)
    return None


def connect(pod_id):
    """Wait for the pod's SSH endpoint, then for sshd itself. Returns (ssh_base, scp_base, host)."""
    info = None
    for i in range(50):
        try:
            j = json.loads(runpodctl("ssh", "info", pod_id, "-o", "json").stdout)
        except Exception:
            j = {"error": "parse"}
        if not j.get("error"):
            info = j
            print(f"ssh ready (~{i * 20}s): {info.get('ip')}:{info.get('port')}", flush=True)
            break
        if i % 3 == 0:
            print(f"[{i * 20}s] {j.get('error')}", flush=True)
        time.sleep(20)
    if not info:
        raise RuntimeError("ssh info never became ready")
    key = (info.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
    host = f"root@{info['ip']}"
    ssh = ["ssh", "-i", key, *SSH_OPTS, "-p", str(info["port"]), host]
    scp = ["scp", "-i", key, *SSH_OPTS, "-P", str(info["port"])]
    for k in range(42):
        if "ok" in subprocess.run(ssh + ["echo ok"], capture_output=True, text=True).stdout:
            print(f"ssh up (~{k * 10}s)", flush=True)
            return ssh, scp, host
        time.sleep(10)
    raise RuntimeError("direct ssh not accepting")


def run_job(ssh, scp, host, model, epochs):
    print("uploading train_on_pod.py + SFT data ...", flush=True)
    with open(POD_SCRIPT) as f:
        subprocess.run(ssh + ["cat > /workspace/train_on_pod.py"], stdin=f, check=True, timeout=60)
    for name in DATA_FILES:
        subprocess.run(scp + [str(RUNS / name), f"{host}:/workspace/{name}"], check=True, timeout=120)

    print("installing deps + running SFT (~30-60 min) ...", flush=True)
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' peft accelerate datasets 2>&1 | tail -2 && "
              "echo '--- RUN ---' && "
              f"read -r HFT && HF_TOKEN=$HFT SFT_MODEL={model} EPOCHS={epochs} python3 train_on_pod.py")
    hf_token = os.environ.get("HF_TOKEN", "")
    run = subprocess.run(ssh + [remote], capture_output=True, text=True, timeout=6600,
                         input=hf_token + "\n")   # the token goes over stdin, never into argv or a log
    (RUNS / "pod_stdout.txt").write_text(run.stdout)
    (RUNS / "pod_stderr.txt").write_text(run.stderr)
    print("=== SFT STDOUT (tail) ===\n" + run.stdout[-3500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-2000:], flush=True)
    fetch = subprocess.run(scp + [f"{host}:/workspace/sft_gen.json", str(RUNS / "sft_gen.json")],
                           capture_output=True, text=True, timeout=300)
    print(f"scp sft_gen.json rc={fetch.returncode} {fetch.stderr[-160:]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", default="3")
    args = ap.parse_args()
    if "RUNPOD_API_KEY" not in os.environ:
        sys.exit("set RUNPOD_API_KEY")
    print(f"model={args.model} epochs={args.epochs}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    pod_id = create_pod()
    if not pod_id:
        sys.exit("no GPU type available")
    (RUNS / "pod_id.txt").write_text(pod_id)
    try:
        ssh, scp, host = connect(pod_id)
        run_job(ssh, scp, host, args.model, args.epochs)
    finally:
        print("terminating pod", pod_id, "...", flush=True)
        print("delete:", runpodctl("pod", "delete", pod_id).stdout[-150:].replace("\n", " "), flush=True)
        print("still listed?:", pod_id in runpodctl("pod", "list").stdout, flush=True)


if __name__ == "__main__":
    main()
