"""The self-cleaning RunPod pattern shared by the probe scripts that rent a GPU: create a pod with a
--terminate-after backstop set at creation, wait until ssh answers, and always delete the pod.
Drives runpodctl with RUNPOD_API_KEY from the environment; the key is never printed or written."""
import datetime
import json
import os
import subprocess
import sys
import time

from generate.common.paths import ROOT

RUNS = ROOT / "runs" / "probe"
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]


def runpodctl(*args, timeout=180):
    env = {**os.environ, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    return subprocess.run(["runpodctl", *args], env=env, capture_output=True, text=True, timeout=timeout)


def _id_and_price(stdout):
    d = json.loads(stdout)
    d = d[0] if isinstance(d, list) else d
    d = d.get("pod") or d
    return d.get("id"), d.get("costPerHr")


def create(name, gpus, disk_gb, hours):
    """Try each GPU type in order. The pod deletes itself after `hours` even if this process dies.
    Returns (pod_id, gpu, usd_per_hour or None); exits when no GPU type is available."""
    term_after = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for gpu in gpus:
        print(f"trying {gpu} (auto-terminate {term_after}) ...", flush=True)
        r = runpodctl("pod", "create", "--name", name, "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
                      "--container-disk-in-gb", str(disk_gb), "--terminate-after", term_after, "-o", "json")
        try:
            pid, price = _id_and_price(r.stdout)
        except Exception:
            pid, price = None, None
        if pid:
            print(f"created on {gpu} -> {pid}", flush=True)
            RUNS.mkdir(parents=True, exist_ok=True)
            (RUNS / f"pod_id_{name}.txt").write_text(pid)   # for manual cleanup if this process is killed
            return pid, gpu, price
        print("  unavailable:", (r.stdout + r.stderr)[-160:].replace("\n", " "), flush=True)
    sys.exit("no GPU type available")


def connect(pid):
    """Wait for the pod's ssh endpoint, then for sshd to answer (big-disk pods start slowly).
    Returns (ssh_prefix, scp_prefix, host)."""
    info = None
    for i in range(50):
        try:
            j = json.loads(runpodctl("ssh", "info", pid, "-o", "json").stdout)
        except Exception:
            j = {"error": "parse"}
        if not j.get("error"):
            info = j
            break
        time.sleep(20)
    if not info:
        raise RuntimeError("ssh info never became ready")
    key = (info.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
    ssh = ["ssh", "-i", key, *SSHOPTS, "-p", str(info["port"]), f"root@{info['ip']}"]
    scp = ["scp", "-i", key, *SSHOPTS, "-P", str(info["port"])]
    for k in range(42):
        if "ok" in subprocess.run(ssh + ["echo ok"], capture_output=True, text=True).stdout:
            print(f"ssh up after ~{k * 10}s", flush=True)
            return ssh, scp, info["ip"]
        time.sleep(10)
    raise RuntimeError("ssh not accepting connections")


def delete(pid):
    print("terminating pod", pid, "...", flush=True)
    print("delete:", runpodctl("pod", "delete", pid).stdout[-150:].replace("\n", " "), flush=True)
    print("still listed?:", pid in runpodctl("pod", "list").stdout, flush=True)
