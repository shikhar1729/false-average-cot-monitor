"""Stage 1 (free, no claude): generic TW-task runner + env-sanity sweep.
Translates each task's Dockerfile -> setup commands on a shared RunPod sandbox, applies the reference
solve.sh, runs the task's own verifier, and records whether the reference PASSES. This measures how many
tasks actually run cleanly in our sandbox BEFORE spending anything on the agent. Self-cleaning pod.

  python scripts/live/env_sanity_sweep.py [N_candidates]
Writes results/scores/env_sanity_sweep.json (list of {task, ok, tail}).
"""
import json, os, re, subprocess, time, datetime, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
RUNS = ROOT / "runs"; RUNS.mkdir(exist_ok=True)   # logs, pod ids, shard stdout (gitignored)
REPO = ROOT / "repo"
TASKS = REPO / "tasks"
ARG = sys.argv[1] if len(sys.argv) > 1 else "40"  # a number (=#candidates) OR a comma-list of task ids
env = os.environ  # keys come from the environment; never printed or written
API = env.get("RUNPOD_API_KEY", "")   # required only when a pod is created
RPENV = {**os.environ, "RUNPOD_API_KEY": API}
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
GPUS = ["NVIDIA A40", "NVIDIA RTX A4000", "NVIDIA A100 80GB PCIe", "NVIDIA RTX A5000"]
SJ = ROOT / "results/scores/env_sanity_sweep.json"
LOG = open(RUNS / "env_sanity_sweep.log", "w")
def log(*a):
    print("[sweep]", *a, flush=True); print("[sweep]", *a, file=LOG, flush=True)


def candidates(n):
    out = []
    for t in sorted(os.listdir(TASKS)):
        base = TASKS / t / "claude-opus-4.6/original_task"
        hs = TASKS / t / "hack_summary.md"
        dock = base / "environment/Dockerfile"
        if not (hs.exists() and dock.exists() and (base / "solution/solve.sh").exists()
                and (base / "tests/test_outputs.py").exists()):
            continue
        if "hollow-implementation" not in hs.read_text(errors="replace"):
            continue
        frm = ""
        for l in dock.read_text(errors="replace").splitlines():
            m = re.match(r"\s*FROM\s+(\S+)", l, re.I)
            if m:
                frm = m.group(1); break
        if not any(frm.startswith(x) for x in ("ubuntu:", "python:", "debian:", "docker.io/ubuntu")):
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def translate(dockerfile_text):
    """Dockerfile -> (workdir, setup_body, env_exports). RUN emitted verbatim (bash handles \\ + heredocs)."""
    lines = dockerfile_text.split("\n")
    setup, envs, workdir = ["mkdir -p /app && cd /app"], [], "/app"
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        if l.lstrip().startswith("#") or not l.strip():
            i += 1; continue
        m = re.match(r"\s*(\w+)\b\s?(.*)", l)
        if not m:
            i += 1; continue
        instr, rest0 = m.group(1).upper(), m.group(2)
        block = [l]
        while block[-1].rstrip().endswith("\\"):   # backslash continuation
            i += 1
            if i >= n:
                break
            block.append(lines[i])
        joined = "\n".join(block)
        hm = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", joined)   # heredoc
        if hm and not any(bl.strip() == hm.group(1) for bl in block[1:]):
            mark = hm.group(1)
            while i + 1 < n:
                i += 1; block.append(lines[i])
                if lines[i].strip() == mark:
                    break
        body = "\n".join([re.match(r"\s*\w+\b\s?(.*)", block[0]).group(1)] + block[1:])
        if instr == "WORKDIR":
            workdir = rest0.strip()
            setup.append(f"mkdir -p {workdir} && cd {workdir}")
        elif instr == "RUN":
            setup.append(body)
        elif instr in ("COPY", "ADD"):
            toks = [t for t in body.split() if not t.startswith("--")]
            if len(toks) >= 2:
                *srcs, dst = toks
                for s in srcs:  # Docker COPY copies dir CONTENTS into dst; a file goes to dst
                    setup.append(f"if [ -d /build/{s} ]; then mkdir -p {dst} && cp -r /build/{s}/. {dst}/; "
                                 f"else mkdir -p $(dirname {dst}) && cp /build/{s} {dst}; fi")
        elif instr == "ENV":
            if "=" in rest0:
                for k, v in re.findall(r"(\w+)=(\"[^\"]*\"|'[^']*'|\S+)", body):
                    envs.append(f"export {k}={v}"); setup.append(f"export {k}={v}")
            else:
                p = rest0.split(None, 1)
                if len(p) == 2:
                    envs.append(f'export {p[0]}="{p[1].strip()}"'); setup.append(f'export {p[0]}="{p[1].strip()}"')
        i += 1
    return workdir, "\n".join(setup), "\n".join(envs)


def main():
    cands = candidates(int(ARG)) if ARG.isdigit() else ARG.split(",")
    log(f"{len(cands)} candidates: {cands}")
    term_after = (datetime.datetime.utcnow() + datetime.timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    def rp(*a, timeout=180):
        return subprocess.run(["runpodctl", *a], env=RPENV, capture_output=True, text=True, timeout=timeout)
    pid = None
    for gpu in GPUS:
        assert API, "RUNPOD_API_KEY is not set (see README Setup)"
        r = rp("pod", "create", "--name", "envsweep", "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
               "--container-disk-in-gb", "40", "--terminate-after", term_after, "-o", "json")
        try:
            pid = json.loads(r.stdout).get("id");
            if pid: log(f"pod {pid} on {gpu}"); break
        except Exception:
            pass
    if not pid:
        log("NO GPU"); return
    open(RUNS / "ENVSWEEP_POD.txt", "w").write(pid)
    results = []
    try:
        info = None
        for _ in range(50):
            try:
                j = json.loads(rp("ssh", "info", pid, "-o", "json").stdout)
            except Exception:
                j = {"error": "x"}
            if not j.get("error"):
                info = j; log(f"ssh {info['ip']}:{info['port']}"); break
            time.sleep(20)
        key = (info.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
        base = ["ssh", "-i", key, *SSHOPTS, "-p", str(info["port"]), f"root@{info['ip']}"]
        scpb = ["scp", "-i", key, *SSHOPTS, "-P", str(info["port"])]

        def sh(cmd, timeout=360):
            for a in range(3):
                try:
                    r = subprocess.run(base + ["bash -s"], input=cmd, capture_output=True, text=True,
                                       errors="replace", timeout=timeout)
                    if r.returncode != 255:
                        return (r.stdout + r.stderr).strip()
                except subprocess.TimeoutExpired:
                    return f"[timeout {timeout}s]"
                time.sleep(6 * (a + 1))
            return ""

        def scp_to(local, remote):
            for a in range(3):
                if subprocess.run(scpb + ["-r", str(local), f"root@{info['ip']}:{remote}"],
                                  capture_output=True, text=True, timeout=200).returncode == 0:
                    return True
                time.sleep(5)
            return False

        for _ in range(48):
            if "ok" in sh("echo ok", timeout=30):
                break
            time.sleep(10)
        log("ssh up")

        for idx, t in enumerate(cands, 1):
            b = TASKS / t / "claude-opus-4.6/original_task"
            try:
                # rm the targets and DO NOT pre-create them: scp -r dir->/target places contents at /target
                sh("rm -rf /build /app /tests /sol /data /home/user; mkdir -p /app", timeout=60)
                ok_scp = (scp_to(b / "environment", "/build") and
                          scp_to(b / "solution", "/sol") and scp_to(b / "tests", "/tests"))
                if not ok_scp:
                    results.append({"task": t, "ok": False, "tail": "scp failed"}); log(f"{idx}/{len(cands)} {t}: scp FAIL"); continue
                wd, setup, envs = translate((b / "environment/Dockerfile").read_text(errors="replace"))
                sh("set -e\n" + setup, timeout=420)
                sh(f"{envs}\ncd {wd} 2>/dev/null || cd /app\nchmod +x /sol/solve.sh 2>/dev/null; bash /sol/solve.sh", timeout=420)
                vout = sh(f"set +e\n{envs}\nexport TEST_DIR=/tests PYTHONPATH={wd}:/app:$PYTHONPATH\ncd {wd} 2>/dev/null || cd /app\n"
                          f"pip install -q --break-system-packages pytest >/dev/null 2>&1 || pip3 install -q pytest >/dev/null 2>&1 || pip install -q pytest >/dev/null 2>&1\n"
                          f"python3 -m pytest /tests/test_outputs.py -q 2>&1 | tail -6", timeout=420)
                ok = bool(re.search(r"\d+ passed", vout)) and not re.search(r"\d+ (failed|error)", vout)
                results.append({"task": t, "ok": ok, "wd": wd, "tail": vout[-200:]})
                log(f"{idx}/{len(cands)} {t}: {'PASS' if ok else 'fail'}  {vout[-90:].strip()}")
            except Exception as e:
                results.append({"task": t, "ok": False, "tail": f"exc {e}"}); log(f"{idx}/{len(cands)} {t}: EXC {e}")
            json.dump(results, open(SJ, "w"), indent=2)
    finally:
        json.dump(results, open(SJ, "w"), indent=2)
        log("delete:", rp("pod", "delete", pid).stdout[-40:]); log("still listed:", pid in rp("pod", "list").stdout)
        okc = [r["task"] for r in results if r.get("ok")]
        log(f"\n==== ENV-SANITY: {len(okc)}/{len(results)} tasks pass ====")
        log("PASSING:", okc)
        log("FAILING:", [r["task"] for r in results if not r.get("ok")])


if __name__ == "__main__":
    main()
