"""Self-cleaning launcher for Exp #1 white-box probe: uploads pod_probe.py + the obf-internals JSONL
to a fresh RunPod GPU, runs the probe, fetches probe_results.json + probe_feats.pt, ALWAYS terminates.
Keys from the environment, direct ssh, --terminate-after backstop.
"""
import json, os, subprocess, time, datetime, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE)); _S = os.path.dirname(HERE); sys.path[:0] = [os.path.join(_S, d) for d in os.listdir(_S) if os.path.isdir(os.path.join(_S, d))] + [os.path.join(ROOT, "monitoring"), os.path.join(ROOT, "repo", "monitoring")]
RUNS = os.path.join(ROOT, "runs"); os.makedirs(RUNS, exist_ok=True)   # logs, pod ids, stdout (gitignored)
GATE = os.path.join(HERE, "pod_probe.py")
DATA = os.environ.get("PROBE_INPUT", os.path.join(ROOT, "results", "internals_inputs_obf.jsonl"))
# argv: [model_id] [tag] [maxtok]  (defaults = the 8B run)
PROBE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
MAXTOK = sys.argv[3] if len(sys.argv) > 3 else "16384"
BIG = "32B" in PROBE_MODEL or "70B" in PROBE_MODEL or "72B" in PROBE_MODEL
env = os.environ  # keys come from the environment; never printed or written
API, HFT = env["RUNPOD_API_KEY"], env.get("HF_TOKEN", "")
RPENV = {**os.environ, "RUNPOD_API_KEY": API}
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
GPUS = (["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"] if BIG
        else ["NVIDIA A40", "NVIDIA RTX A5000", "NVIDIA RTX 4090", "NVIDIA RTX A4000", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"])
print(f"model={PROBE_MODEL} tag={TAG!r} maxtok={MAXTOK} gpus={GPUS[0]}...", flush=True)


def rp(*a, timeout=180):
    return subprocess.run(["runpodctl", *a], env=RPENV, capture_output=True, text=True, timeout=timeout)


def extract_id(s):
    d = json.loads(s)
    if isinstance(d, dict):
        return d.get("id") or (d.get("pod") or {}).get("id")
    if isinstance(d, list) and d:
        return d[0]["id"]
    raise KeyError("no id")


term_after = (datetime.datetime.utcnow() + datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
pid = None
for gpu in GPUS:
    print(f"trying {gpu} (auto-terminate {term_after}) ...", flush=True)
    r = rp("pod", "create", "--name", "probe", "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
           "--container-disk-in-gb", "160" if BIG else "60", "--terminate-after", term_after, "-o", "json")
    try:
        pid = extract_id(r.stdout); print(f"created on {gpu} -> {pid}", flush=True); break
    except Exception:
        print("  unavailable:", (r.stdout + r.stderr)[-160:].replace("\n", " "), flush=True)
if not pid:
    print("ALL GPU OPTIONS UNAVAILABLE"); sys.exit(1)
open(os.path.join(RUNS, "PROBE_POD_ID.txt"), "w").write(pid)

try:
    info = None
    for i in range(50):
        try:
            j = json.loads(rp("ssh", "info", pid, "-o", "json").stdout)
        except Exception:
            j = {"error": "parse"}
        if not j.get("error"):
            info = j; print(f"ssh ready (~{i*20}s): {info.get('ip')}:{info.get('port')}", flush=True); break
        if i % 3 == 0:
            print(f"[{i*20}s] {j.get('error')}", flush=True)
        time.sleep(20)
    if not info:
        raise RuntimeError("ssh info never became ready")
    keypath = (info.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
    base = ["ssh", "-i", keypath, *SSHOPTS, "-p", str(info["port"]), f"root@{info['ip']}"]
    scpbase = ["scp", "-i", keypath, *SSHOPTS, "-P", str(info["port"])]

    up = False
    for k in range(42):  # sshd warmup, ~7 min (big-disk pods init slowly)
        if "ok" in subprocess.run(base + ["echo ok"], capture_output=True, text=True).stdout:
            up = True; print(f"ssh up (~{k*10}s)", flush=True); break
        time.sleep(10)
    if not up:
        raise RuntimeError("direct ssh not accepting")

    print("uploading pod_probe.py + input JSONL ...", flush=True)
    with open(GATE) as f:
        subprocess.run(base + ["cat > /workspace/pod_probe.py"], stdin=f, check=True, timeout=60)
    subprocess.run(scpbase + [os.path.abspath(DATA), f"root@{info['ip']}:/workspace/internals_inputs_obf.jsonl"],
                   check=True, timeout=300)

    print("installing deps + running probe (~30-50 min) ...", flush=True)
    VERBAL = os.environ.get("VERBAL", "1")
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' accelerate scikit-learn safetensors "
              "huggingface_hub 2>&1 | tail -2 && echo '--- RUN ---' && "
              f"read -r HFT && HF_TOKEN=$HFT PROBE_MODEL={PROBE_MODEL} MAXTOK={MAXTOK} VERBAL={VERBAL} python3 pod_probe.py")
    run = subprocess.run(base + [remote], capture_output=True, text=True, timeout=6000, input=HFT + "\n")   # token via stdin, not argv
    open(os.path.join(RUNS, f"pod_probe{TAG}_stdout.txt"), "w").write(run.stdout)
    open(os.path.join(RUNS, f"pod_probe{TAG}_stderr.txt"), "w").write(run.stderr)
    print("=== PROBE STDOUT (tail) ===\n" + run.stdout[-3500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-1500:], flush=True)
    print("fetching results ...", flush=True)
    for art, local in (("probe_results.json", f"probe_results{TAG}.json"), ("probe_feats.pt", f"probe_feats{TAG}.pt")):
        fr = subprocess.run(scpbase + [f"root@{info['ip']}:/workspace/{art}", os.path.join(HERE, local)],
                            capture_output=True, text=True, timeout=300)
        print(f"scp {art}->{local} rc={fr.returncode} {fr.stderr[-120:]}", flush=True)
finally:
    print("terminating pod", pid, "...", flush=True)
    print("delete:", rp("pod", "delete", pid).stdout[-150:].replace("\n", " "), flush=True)
    print("still listed?:", pid in rp("pod", "list").stdout, flush=True)
