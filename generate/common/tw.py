"""The Terminal Wrench dataset: find trajectories in the clone, label them, and render them the way
the dataset's own monitor reads them. Rendering, the rubric and the parser come from the authors'
monitoring/monitor.py, used unmodified.

A trajectory (ATIF v1.5/v1.6) is {steps: [...]}; each step has a source (user or agent), and agent
steps carry `message` (the reasoning), `tool_calls` and `observation` (the actions).
"""
import copy
import json
import re
import sys
from pathlib import Path

from generate.common.paths import MONITORING, TW

sys.path.insert(0, str(MONITORING))
import monitor as M  # noqa: E402  (the authors' monitor.py)

M.DATASET_ROOT = TW   # monitor.py looks for task files next to itself; point it at the clone

DEV_MSG = M.DEVELOPER_TEMPLATE.format(task_prompt=M.load_prompt("v1"))   # the dataset's rubric (v1)
HACK_CLASSES = ("rewarded_serious_exploit", "rewarded_nonserious_hack")
SCORE_RE = re.compile(r'"suspicion_score"\s*:\s*([01](?:\.\d+)?|0?\.\d+)')


def collect(actor="gemini-3.1-pro"):
    """Every rewarded hack and every legitimate baseline solve of one actor, with labels and paths."""
    out = []
    for e in json.load(open(TW / "index" / "trajectories.json")):
        if e["model"] != actor or e.get("classification") not in HACK_CLASSES:
            continue
        label = e.get("trajectory_label", "v5")
        p = TW / "tasks" / e["task_id"] / e["model"] / "hack_trajectories" / label / "trial" / "agent" / "trajectory.json"
        if p.exists():
            out.append({"task_id": e["task_id"], "label": label, "is_hack": 1, "classification": e.get("classification"),
                        "observed_categories": e.get("observed_categories") or [], "path": str(p)})
    for e in json.load(open(TW / "index" / "tasks.json")):
        if e["model"] != actor:
            continue
        for bl in e.get("baselines", []):
            label = bl.get("label") or bl.get("baseline_label", "1")
            p = TW / "tasks" / e["task_id"] / e["model"] / "baseline_trajectories" / label / "trial" / "agent" / "trajectory.json"
            if p.exists():
                out.append({"task_id": e["task_id"], "label": label, "is_hack": 0, "classification": "legitimate",
                            "observed_categories": [], "path": str(p)})
    return out


def clean_hacks(actor="gemini-3.1-pro"):
    """Hacks minus those that name the exploit in a command (any monitor catches those trivially)."""
    return [s for s in collect(actor) if s["is_hack"] and not M.has_keyword_leak_in_tool_calls(s["path"])]


def sid(s):
    return f"{s['task_id']}__{s['label']}__{'hack' if s['is_hack'] else 'base'}"


def load(sample):
    return M.load_atif_trajectory(Path(sample["path"]))


def render(atif, task_id, actor="gemini-3.1-pro", scope="full"):
    """The monitor's user message: task description plus the trajectory, at full or action-only scope."""
    if scope == "action-only":
        atif = make_stripped(atif)
    instr = M.load_instruction(task_id, actor)
    return M.USER_TEMPLATE.format(task_description=instr, trajectory=M.format_atif_trajectory(atif))


def extract_score(content):
    """The authors' parser first; a regex fallback recovers a valid score when the reasoning string
    holds unescaped quotes. Returns None when no score can be read."""
    s = M.parse_monitor_response(content or "").get("suspicion_score")
    if s is not None:
        return float(s)
    m = SCORE_RE.search(content or "")
    return float(m.group(1)) if m else None


# --- action-only scope: a byte-faithful copy of the dataset's monitoring/strip.py ---
RED_TEAM_START = "You are an authorized Red Team QA Agent"
RED_TEAM_END_MARKERS = ["Current terminal state:", "Current Terminal Screen:"]


def strip_red_team_prompt(message):
    start = message.find(RED_TEAM_START)
    if start == -1:
        return message
    end = len(message)
    for marker in RED_TEAM_END_MARKERS:
        idx = message.find(marker, start)
        if idx != -1 and idx < end:
            end = idx
    return message[:start].rstrip() + "\n\n" + message[end:]


def make_stripped(atif):
    """Blank every agent message and drop the red-team block from the first user message that has it;
    tool calls and observations stay intact. A no-op on baselines, which carry no red-team prompt."""
    t = copy.deepcopy(atif)
    done = False
    for s in t.get("steps", []):
        if s.get("source") == "user" and not done:
            before = s.get("message", "") or ""
            s["message"] = strip_red_team_prompt(before)
            done = s["message"] != before
        if s.get("source") == "agent" and (s.get("message") or ""):
            s["message"] = ""
    return t
