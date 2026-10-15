"""Helpers shared by the live scripts: read a Terminal Wrench task, turn its Dockerfile into shell setup
commands for a plain pod, build its verifier command, render a live trajectory for the monitor, and
create, reach and delete a self-cleaning RunPod pod (via runpodctl and RUNPOD_API_KEY)."""
import datetime
import json
import os
import re
import subprocess
import time

from generate.common.paths import TW

TASKS = TW / "tasks"
SSHOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]


def task_dir(task):
    return TASKS / task / "claude-opus-4.6/original_task"


def translate(dockerfile_text):
    """Dockerfile -> (workdir, setup_body, env_exports). RUN bodies are emitted verbatim (bash handles
    backslash continuations and heredocs); COPY/ADD read from the task's environment dir scp'd to /build."""
    lines = dockerfile_text.split("\n")
    setup, envs, workdir = ["mkdir -p /app && cd /app"], [], "/app"
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("#") or not line.strip():
            i += 1
            continue
        m = re.match(r"\s*(\w+)\b\s?(.*)", line)
        if not m:
            i += 1
            continue
        instr, rest0 = m.group(1).upper(), m.group(2)
        block = [line]
        while block[-1].rstrip().endswith("\\"):   # backslash continuation
            i += 1
            if i >= n:
                break
            block.append(lines[i])
        joined = "\n".join(block)
        hm = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", joined)   # heredoc: read on to its end marker
        if hm and not any(bl.strip() == hm.group(1) for bl in block[1:]):
            mark = hm.group(1)
            while i + 1 < n:
                i += 1
                block.append(lines[i])
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
                for s in srcs:  # Docker COPY copies a dir's contents into dst; a file goes to dst
                    setup.append(f"if [ -d /build/{s} ]; then mkdir -p {dst} && cp -r /build/{s}/. {dst}/; "
                                 f"else mkdir -p $(dirname {dst}) && cp /build/{s} {dst}; fi")
        elif instr == "ENV":
            if "=" in rest0:
                for k, v in re.findall(r"(\w+)=(\"[^\"]*\"|'[^']*'|\S+)", body):
                    envs.append(f"export {k}={v}")
                    setup.append(f"export {k}={v}")
            else:
                p = rest0.split(None, 1)
                if len(p) == 2:
                    envs.append(f'export {p[0]}="{p[1].strip()}"')
                    setup.append(f'export {p[0]}="{p[1].strip()}"')
        i += 1
    return workdir, "\n".join(setup), "\n".join(envs)


def verify_command(wd, envs):
    """The task's own pytest verifier, run from its working dir."""
    return (f"set +e\n{envs}\nexport TEST_DIR=/tests PYTHONPATH={wd}:/app:$PYTHONPATH\ncd {wd} 2>/dev/null || cd /app\n"
            f"pip install -q --break-system-packages pytest >/dev/null 2>&1 || pip3 install -q pytest >/dev/null 2>&1 || pip install -q pytest >/dev/null 2>&1\n"
            f"python3 -m pytest /tests/test_outputs.py -q 2>&1 | tail -6")


def solve_command(wd, envs):
    """Apply the reference solution scp'd to /sol."""
    return f"{envs}\ncd {wd} 2>/dev/null || cd /app\nchmod +x /sol/solve.sh 2>/dev/null; bash /sol/solve.sh"


def passed(vout):
    return bool(re.search(r"\d+ passed", vout)) and not re.search(r"\d+ (failed|error)", vout)


def render(steps, mode):
    """A live trajectory as the monitor reads it. mode picks the reasoning shown per step: 'emit' (the
    narration the agent wrote), 'hidden' (the provider's reasoning field) or 'blank' (action-only)."""
    return "\n\n".join(f"[AGENT]\n{ {'emit': s['emit'], 'hidden': s['hidden'], 'blank': '(reasoning withheld)'}[mode] }\n"
                       f"[ACTIONS]\n{s['command']}\n[OBSERVATION]\n{s['observation']}" for s in steps)


# ---------------- the self-cleaning pod: create, reach, delete ----------------
def runpodctl(*args, timeout=180):
    env = {**os.environ, "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", "")}
    return subprocess.run(["runpodctl", *args], env=env, capture_output=True, text=True, timeout=timeout)


def create_pod(name, gpus, disk_gb, hours, log):
    """Try each GPU type in order. The --terminate-after backstop is set at creation, so the pod dies
    even if this process does. Returns the pod id, or None when no GPU type is available."""
    term_after = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for gpu in gpus:
        assert os.environ.get("RUNPOD_API_KEY"), "RUNPOD_API_KEY is not set"
        r = runpodctl("pod", "create", "--name", name, "--gpu-id", gpu, "--template-id", "runpod-torch-v240",
                      "--container-disk-in-gb", str(disk_gb), "--terminate-after", term_after, "-o", "json")
        try:
            pid = json.loads(r.stdout).get("id")
        except Exception:
            continue
        if pid:
            log(f"pod {pid} on {gpu}")
            return pid
    log("NO GPU")
    return None


def ssh_info(pid, log):
    """Poll until the pod reports its ssh endpoint (up to about 17 minutes)."""
    for _ in range(50):
        try:
            j = json.loads(runpodctl("ssh", "info", pid, "-o", "json").stdout)
        except Exception:
            j = {"error": "x"}
        if not j.get("error"):
            log(f"ssh {j['ip']}:{j['port']}")
            return j
        time.sleep(20)
    raise RuntimeError(f"pod {pid} never reported ssh info")


def ssh_bases(info, extra_opts=()):
    """The ssh and scp command prefixes for this pod, using the runpodctl key."""
    key = (info.get("ssh_key") or {}).get("path") or os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
    ssh = ["ssh", "-i", key, *SSHOPTS, *extra_opts, "-p", str(info["port"]), f"root@{info['ip']}"]
    scp = ["scp", "-i", key, *SSHOPTS, *extra_opts, "-P", str(info["port"])]
    return ssh, scp


def scp_to(scp, ip, local, remote):
    for _ in range(3):
        if subprocess.run(scp + ["-r", str(local), f"root@{ip}:{remote}"],
                          capture_output=True, text=True, timeout=200).returncode == 0:
            return True
        time.sleep(5)
    return False


def delete_pod(pid, log):
    log("delete:", runpodctl("pod", "delete", pid).stdout[-40:])
    log("still listed:", pid in runpodctl("pod", "list").stdout)
