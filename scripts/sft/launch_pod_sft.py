"""Self-cleaning launcher for the SFT-generalization experiment: uploads pod_sft.py + the SFT data,
LoRA-SFTs Qwen3-8B on obfuscated reasoning, generates held-out reasoning (base vs SFT'd), fetches
sft_gen.json, ALWAYS terminates. Mirrors launch_pod_probe.py.
"""
import json, os, subprocess, time, datetime, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE)); _S = os.path.dirname(HERE); sys.path[:0] = [os.path.join(_S, d) for d in os.listdir(_S) if os.path.isdir(os.path.join(_S, d))] + [os.path.join(ROOT, "monitoring"), os.path.join(ROOT, "repo", "monitoring")]
RUNS = os.path.join(ROOT, "runs"); os.makedirs(RUNS, exist_ok=True)   # logs, pod ids, stdout (gitignored)
RES = os.path.join(ROOT, "results")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
EPOCHS = sys.argv[2] if len(sys.argv) > 2 else "3"
env = os.environ  # keys come from the environment; never printed or written
API, HFT = env["RUNPOD_API_KEY"], env.get("HF_TOKEN", "")
RPENV = {**os.environ, "RUNPOD_API_KEY": API}
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA H100 80GB HBM3"]
print(f"model={MODEL} epochs={EPOCHS}", flush=True)


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
    r = rp("pod", "create", "--name", "sft-gen", "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
           "--container-disk-in-gb", "100", "--terminate-after", term_after, "-o", "json")
    try:
        pid = extract_id(r.stdout); print(f"created on {gpu} -> {pid}", flush=True); break
    except Exception:
        print("  unavailable:", (r.stdout + r.stderr)[-160:].replace("\n", " "), flush=True)
if not pid:
    print("ALL GPU OPTIONS UNAVAILABLE"); sys.exit(1)
open(os.path.join(RUNS, "SFT_POD_ID.txt"), "w").write(pid)

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
    for k in range(42):
        if "ok" in subprocess.run(base + ["echo ok"], capture_output=True, text=True).stdout:
            up = True; print(f"ssh up (~{k*10}s)", flush=True); break
        time.sleep(10)
    if not up:
        raise RuntimeError("direct ssh not accepting")

    print("uploading pod_sft.py + SFT data ...", flush=True)
    with open(os.path.join(HERE, "pod_sft.py")) as f:
        subprocess.run(base + ["cat > /workspace/pod_sft.py"], stdin=f, check=True, timeout=60)
    for fn in ("sft_train.jsonl", "sft_heldout.jsonl"):
        subprocess.run(scpbase + [os.path.join(RES, fn), f"root@{info['ip']}:/workspace/{fn}"], check=True, timeout=120)

    print("installing deps + running SFT (~30-60 min) ...", flush=True)
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' peft accelerate datasets 2>&1 | tail -2 && "
              "echo '--- RUN ---' && "
              f"read -r HFT && HF_TOKEN=$HFT SFT_MODEL={MODEL} EPOCHS={EPOCHS} python3 pod_sft.py")
    run = subprocess.run(base + [remote], capture_output=True, text=True, timeout=6600, input=HFT + "\n")   # token via stdin, not argv
    open(os.path.join(RUNS, "pod_sft_stdout.txt"), "w").write(run.stdout)
    open(os.path.join(RUNS, "pod_sft_stderr.txt"), "w").write(run.stderr)
    print("=== SFT STDOUT (tail) ===\n" + run.stdout[-3500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-2000:], flush=True)
    fr = subprocess.run(scpbase + [f"root@{info['ip']}:/workspace/sft_gen.json", os.path.join(RES, "sft_gen.json")],
                        capture_output=True, text=True, timeout=300)
    print(f"scp sft_gen.json rc={fr.returncode} {fr.stderr[-160:]}", flush=True)
finally:
    print("terminating pod", pid, "...", flush=True)
    print("delete:", rp("pod", "delete", pid).stdout[-150:].replace("\n", " "), flush=True)
    print("still listed?:", pid in rp("pod", "list").stdout, flush=True)
