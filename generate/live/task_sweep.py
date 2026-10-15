"""Which Terminal Wrench tasks run cleanly in our pod sandbox (Section 5: 27 of 44 candidates). For each
candidate, translate its Dockerfile into setup commands on one shared pod, apply the reference solve.sh
and run the task's own verifier; a task is usable when the reference passes. No model is called.
Writes data/live/task_sweep.json (one {task, ok, wd, tail} row per candidate).
  python -m generate.live.task_sweep               # the first 44 candidates (or give a count or a comma list of task ids)
  python -m generate.live.task_sweep --dry-run     # $0: list the candidates, create no pod
Needs the TW clone and RUNPOD_API_KEY (runpodctl); a few dollars of pod time, no API cost.
"""
import argparse
import json
import os
import re
import subprocess
import time

from generate.common.paths import DATA, ROOT
from generate.live import sandbox

OUT = DATA / "live" / "task_sweep.json"
RUNS = ROOT / "runs" / "live"
GPUS = ["NVIDIA A40", "NVIDIA RTX A4000", "NVIDIA A100 80GB PCIe", "NVIDIA RTX A5000"]
BASE_IMAGES = ("ubuntu:", "python:", "debian:", "docker.io/ubuntu")
LOG = None
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")   # ssh prints the pod's address in its host-key warning


def log(*a):
    print("[sweep]", *a, flush=True)
    if LOG:
        print("[sweep]", *a, file=LOG, flush=True)


def candidates(n):
    """The first n tasks (sorted by id) with a hollow-implementation hack, a reference solution, a pytest
    verifier and a Dockerfile on a plain Ubuntu, Debian or Python base image."""
    out = []
    for t in sorted(os.listdir(sandbox.TASKS)):
        base = sandbox.task_dir(t)
        hs = sandbox.TASKS / t / "hack_summary.md"
        dock = base / "environment/Dockerfile"
        if not (hs.exists() and dock.exists() and (base / "solution/solve.sh").exists()
                and (base / "tests/test_outputs.py").exists()):
            continue
        if "hollow-implementation" not in hs.read_text(errors="replace"):
            continue
        frm = ""
        for line in dock.read_text(errors="replace").splitlines():
            m = re.match(r"\s*FROM\s+(\S+)", line, re.I)
            if m:
                frm = m.group(1)
                break
        if not any(frm.startswith(x) for x in BASE_IMAGES):
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def connect(info):
    """sh(cmd) and scp(local, remote) bound to this pod."""
    ssh, scp = sandbox.ssh_bases(info)

    def sh(cmd, timeout=360):
        for a in range(3):
            try:
                r = subprocess.run(ssh + ["bash -s"], input=cmd, capture_output=True, text=True,
                                   errors="replace", timeout=timeout)
                if r.returncode != 255:
                    return (r.stdout + r.stderr).strip()
            except subprocess.TimeoutExpired:
                return f"[timeout {timeout}s]"
            time.sleep(6 * (a + 1))
        return ""

    def scp_to(local, remote):
        return sandbox.scp_to(scp, info["ip"], local, remote)

    return sh, scp_to


def check_task(sh, scp_to, task):
    """Set the task up from scratch, apply the reference, run the verifier."""
    b = sandbox.task_dir(task)
    # remove the targets and do not pre-create them: scp -r dir -> /target places the contents at /target
    sh("rm -rf /build /app /tests /sol /data /home/user; mkdir -p /app", timeout=60)
    if not (scp_to(b / "environment", "/build") and scp_to(b / "solution", "/sol") and scp_to(b / "tests", "/tests")):
        return {"task": task, "ok": False, "tail": "scp failed"}
    wd, setup, envs = sandbox.translate((b / "environment/Dockerfile").read_text(errors="replace"))
    sh("set -e\n" + setup, timeout=420)
    sh(sandbox.solve_command(wd, envs), timeout=420)
    vout = sh(sandbox.verify_command(wd, envs), timeout=420)
    return {"task": task, "ok": sandbox.passed(vout), "wd": wd, "tail": IPV4.sub("<redacted>", vout[-200:])}


def sweep(tasks):
    pid = sandbox.create_pod("envsweep", GPUS, disk_gb=40, hours=4, log=log)
    if not pid:
        return
    (RUNS / "task_sweep_pod.txt").write_text(pid)
    results = []
    try:
        sh, scp_to = connect(sandbox.ssh_info(pid, log))
        for _ in range(48):
            if "ok" in sh("echo ok", timeout=30):
                break
            time.sleep(10)
        log("ssh up")
        for idx, t in enumerate(tasks, 1):
            try:
                row = check_task(sh, scp_to, t)
            except Exception as e:
                row = {"task": t, "ok": False, "tail": f"exc {e}"}
            results.append(row)
            log(f"{idx}/{len(tasks)} {t}: {'PASS' if row['ok'] else 'fail'}  {row['tail'][-90:].strip()}")
            OUT.write_text(json.dumps(results, indent=1))
    finally:
        OUT.write_text(json.dumps(results, indent=1))
        sandbox.delete_pod(pid, log)
        log(f"{sum(r['ok'] for r in results)}/{len(results)} tasks pass:", [r["task"] for r in results if r["ok"]])


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="?", default="44", help="number of candidates, or a comma list of task ids")
    ap.add_argument("--dry-run", action="store_true", help="$0: list the candidates; no pod")
    a = ap.parse_args()
    tasks = candidates(int(a.tasks)) if a.tasks.isdigit() else a.tasks.split(",")
    if a.dry_run:
        print(f"{len(tasks)} candidates: {tasks}; no pod created")
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG = open(RUNS / "task_sweep.log", "w")
    log(f"{len(tasks)} candidates: {tasks}")
    sweep(tasks)


if __name__ == "__main__":
    main()
