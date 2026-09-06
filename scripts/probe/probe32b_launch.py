"""32B-probe self-cleaning launcher (mirrors launch_pod_probe.py; own pod-id file so it cannot collide with a
concurrent probe launch). Uploads probe32b_pod.py + the inputs + directions, runs, fetches probe32b_out + probe32b_feats
into results/appendix/probe_32b/ (also on failure), ALWAYS terminates.

  .venv/bin/python scripts/probe32b_launch.py                                   # 8B run (original files)
  .venv/bin/python scripts/probe32b_launch.py --tag 32B_A --model Qwen/Qwen3-32B --cap2 6144 --cost-cap 2.9

--tag T reads probe32b_inputs_T.jsonl / probe32b_dirs_<T-without-_A/_B>.pt and writes probe32b_out_T.jsonl / probe32b_feats_T.pt /
pod_stdout_T.txt / pod_stderr_T.txt / POD_ID_T.txt under runs/; --cost-cap (USD) sets the pod's RUN_SECONDS deadline from the GPU's
hourly price so the run stops, is fetched and deleted inside the budget; --terminate-after backstop is
creation + --backstop-h (<= 3 h).
Secrets: the HF token is written to /workspace/.hf_env over ssh STDIN (mode 600) and sourced by the remote
shell; it never appears on a remote command line or in any local file.
"""
import argparse, datetime, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--cap2", default="0")
ap.add_argument("--cost-cap", type=float, default=0.0, help="USD; 0 = no deadline")
ap.add_argument("--backstop-h", type=float, default=3.0)
ap.add_argument("--disk-gb", default="", help="container disk; default 160 for 32B (bf16 weights ~65 GB), else 60")
A = ap.parse_args()
assert A.backstop_h <= 3.0
A.disk_gb = A.disk_gb or ("160" if "32B" in A.model else "60")
sfx = f"_{A.tag}" if A.tag else ""
dsfx = ("_" + A.tag.split("_")[0]) if A.tag else ""       # 32B_A / 32B_B share probe32b_dirs_32B.pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE)); _S = os.path.dirname(HERE); sys.path[:0] = [os.path.join(_S, d) for d in os.listdir(_S) if os.path.isdir(os.path.join(_S, d))] + [os.path.join(ROOT, "monitoring"), os.path.join(ROOT, "repo", "monitoring")]
OUT_DIR = os.path.join(ROOT, "results", "appendix", "probe_32b")
env = os.environ  # keys come from the environment; never printed or written
RPENV = {**os.environ, "RUNPOD_API_KEY": env["RUNPOD_API_KEY"]}
HFT = env.get("HF_TOKEN", "")
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
if "32B" not in A.model:
    GPUS += ["NVIDIA A40", "NVIDIA RTX A6000"]
PRICE = {"NVIDIA A100 80GB PCIe": 1.64, "NVIDIA A100-SXM4-80GB": 1.99, "NVIDIA H100 80GB HBM3": 2.99, "NVIDIA H100 PCIe": 2.69,
         "NVIDIA A40": 0.44, "NVIDIA RTX A6000": 0.49}      # conservative fallbacks if the API omits costPerHr
for f in (f"probe32b_inputs{sfx}.jsonl", f"probe32b_dirs{dsfx}.pt"):
    assert os.path.exists(os.path.join(OUT_DIR, f)), f


def rp(*a, timeout=180):
    return subprocess.run(["runpodctl", *a], env=RPENV, capture_output=True, text=True, timeout=timeout)


def extract(s):
    d = json.loads(s)
    d = d[0] if isinstance(d, list) else d
    d = d.get("pod") or d
    return d.get("id"), d.get("costPerHr")


t0 = time.time()
term_after = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=A.backstop_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
pid = None
for gpu in GPUS:
    print(f"trying {gpu} (auto-terminate {term_after}) ...", flush=True)
    r = rp("pod", "create", "--name", f"probe32b{sfx}", "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
           "--container-disk-in-gb", A.disk_gb, "--terminate-after", term_after, "-o", "json")
    try:
        pid, price = extract(r.stdout)
        assert pid
        price = float(price) if price else PRICE[gpu]
        print(f"created on {gpu} -> {pid}  ({price:.2f} USD/h)", flush=True); break
    except Exception:
        print("  unavailable:", (r.stdout + r.stderr)[-160:].replace("\n", " "), flush=True)
if not pid:
    print("ALL GPU OPTIONS UNAVAILABLE"); sys.exit(1)
open(os.path.join(OUT_DIR, f"POD_ID{sfx}.txt"), "w").write(pid)
hours_cap = A.cost_cap / price if A.cost_cap else A.backstop_h

info = None
try:
    for i in range(50):
        try:
            j = json.loads(rp("ssh", "info", pid, "-o", "json").stdout)
        except Exception:
            j = {"error": "parse"}
        if not j.get("error"):
            info = j; print(f"ssh ready (~{i*20}s): {info.get('ip')}:{info.get('port')}", flush=True); break
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
    print("uploading ...", flush=True)
    subprocess.run(scpbase + [os.path.join(OUT_DIR, f"probe32b_inputs{sfx}.jsonl"), f"root@{info['ip']}:/workspace/probe32b_inputs.jsonl"], check=True, timeout=300)
    subprocess.run(scpbase + [os.path.join(OUT_DIR, f"probe32b_dirs{dsfx}.pt"), f"root@{info['ip']}:/workspace/probe32b_dirs.pt"], check=True, timeout=300)
    subprocess.run(scpbase + [os.path.join(HERE, "probe32b_pod.py"), f"root@{info['ip']}:/workspace/probe32b_pod.py"], check=True, timeout=60)
    # secrets over stdin, never on a command line
    subprocess.run(base + ["umask 077 && cat > /workspace/.hf_env"], input=f"HF_TOKEN={HFT}\n", text=True, check=True, timeout=60)
    run_seconds = max(600, int(hours_cap * 3600 - (time.time() - t0) - 480))   # leave 8 min to fetch + delete
    print(f"installing deps + running (deadline {run_seconds/3600:.2f} h on the pod, backstop {term_after}) ...", flush=True)
    remote = ("cd /workspace && pip -q install 'transformers==4.55.4' accelerate safetensors huggingface_hub 2>&1 | tail -1 && "
              f"echo '--- RUN ---' && set -a && . /workspace/.hf_env && set +a && "
              f"PROBE_MODEL={A.model} CAP2={A.cap2} RUN_SECONDS={run_seconds} python3 probe32b_pod.py; rc=$?; rm -f /workspace/.hf_env; exit $rc")
    run = subprocess.run(base + [remote], capture_output=True, text=True, timeout=int(run_seconds + 1500))
    open(os.path.join(OUT_DIR, f"pod_stdout{sfx}.txt"), "w").write(run.stdout)
    open(os.path.join(OUT_DIR, f"pod_stderr{sfx}.txt"), "w").write(run.stderr)
    print("=== STDOUT (tail) ===\n" + run.stdout[-2500:], flush=True)
    if run.returncode != 0:
        print("=== STDERR (tail) ===\n" + run.stderr[-1500:], flush=True)
finally:
    try:
        if info:
            for art in ("probe32b_out.jsonl", "probe32b_feats.pt"):
                fr = subprocess.run(scpbase + [f"root@{info['ip']}:/workspace/{art}", os.path.join(OUT_DIR, art.replace(".", f"{sfx}.", 1))],
                                    capture_output=True, text=True, timeout=600)
                print(f"scp {art} rc={fr.returncode} {fr.stderr[-120:]}", flush=True)
    except Exception as e:
        print("fetch failed:", e, flush=True)
    print("terminating pod", pid, "...", flush=True)
    print("delete:", rp("pod", "delete", pid).stdout[-150:].replace("\n", " "), flush=True)
    print("still listed?:", pid in rp("pod", "list").stdout, flush=True)
    print(f"elapsed {(time.time()-t0)/3600:.2f} h, ~{(time.time()-t0)/3600*price:.2f} USD", flush=True)
