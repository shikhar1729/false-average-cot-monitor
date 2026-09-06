"""Monitor-cost estimate for the Gemini-3.1-pro pool.

Reuses the OFFICIAL monitor I/O (monitoring/monitor.py): same developer+user
templates, same trajectory renderer (format_atif_trajectory, which skips user
steps), same tiktoken counter, same MODEL_PRICING. Stripped inputs use our
byte-validated tw_lib.make_stripped (official stripped baselines are NOT shipped,
only stripped hacks — this is the gap our stripper fills). NO API CALLS.

Scoring model (per trajectory):
  M0 (patcher)   scores FULL scope      -> picks which hacks are "caught"
  M1, M2 (held)  score STRIPPED scope   -> monitorability of survivors
Baselines are scored under each monitor at its scope for FPR calibration.
We report the primary plan plus an all-monitors-x-both-scopes upper bound.
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
REPO = ROOT / "repo"
RESULTS = ROOT / "results"
sys.path.insert(0, str(REPO / "monitoring"))

import tw_lib as tw
import monitor as M  # official harness: templates, renderer, counter, pricing

POOL_MODEL = "gemini-3.1-pro"
PROMPT = "v1"
DEV_PREFIX = M.DEVELOPER_TEMPLATE.format(task_prompt=M.load_prompt(PROMPT))
DEV_TOKENS = M.count_tokens(DEV_PREFIX)  # constant per call
EST_OUT = M.EST_OUTPUT_TOKENS_PER_SAMPLE  # 300

# candidate monitors (name -> ($/M in, $/M out)); cross-family priced from a generic band
CANDIDATES = {
    "gpt-5.4-nano": (0.10, 0.40),   # M0 replica of authors' monitor
    "gpt-5.4-mini": (0.30, 1.20),
    "frontier~$1/M": (1.00, 4.00),  # stand-in for a Claude/Gemini held-out monitor
    "frontier~$3/M": (3.00, 12.00),
}


def input_tokens(atif, instruction, stripped):
    a = tw.make_stripped(atif) if stripped else atif
    user_msg = M.USER_TEMPLATE.format(
        task_description=instruction, trajectory=M.format_atif_trajectory(a))
    return DEV_TOKENS + M.count_tokens(user_msg)


def gemini_samples():
    """Yield (atif, instruction, klass) for every Gemini hack + baseline."""
    # hacks: index/trajectories.json, serious + nonserious
    entries = json.load(open(REPO / "index" / "trajectories.json"))
    for e in entries:
        if e["model"] != POOL_MODEL:
            continue
        if e.get("classification") not in ("rewarded_serious_exploit", "rewarded_nonserious_hack"):
            continue
        label = e.get("trajectory_label", "v5")
        p = REPO / "tasks" / e["task_id"] / e["model"] / "hack_trajectories" / label / "trial" / "agent" / "trajectory.json"
        if p.exists():
            yield M.load_atif_trajectory(p), M.load_instruction(e["task_id"], e["model"]), "hack"
    # baselines: index/tasks.json
    tasks = json.load(open(REPO / "index" / "tasks.json"))
    for e in tasks:
        if e["model"] != POOL_MODEL:
            continue
        for bl in e.get("baselines", []):
            label = bl.get("label") or bl.get("baseline_label", "1")
            p = REPO / "tasks" / e["task_id"] / e["model"] / "baseline_trajectories" / label / "trial" / "agent" / "trajectory.json"
            if p.exists():
                yield M.load_atif_trajectory(p), M.load_instruction(e["task_id"], e["model"]), "baseline"


def main():
    RESULTS.mkdir(exist_ok=True)
    n_hack = n_base = 0
    full_hack = full_base = strip_hack = strip_base = 0
    sample_rows = []  # a few rendered monitor inputs for eyeballing
    seen = {"hack": 0, "baseline": 0}
    for atif, instr, klass in gemini_samples():
        if atif is None:
            continue
        ft = input_tokens(atif, instr, stripped=False)
        st = input_tokens(atif, instr, stripped=True)
        if klass == "hack":
            n_hack += 1; full_hack += ft; strip_hack += st
        else:
            n_base += 1; full_base += ft; strip_base += st
        if seen[klass] < 3:
            seen[klass] += 1
            for stripped in (False, True):
                a = tw.make_stripped(atif) if stripped else atif
                sample_rows.append({
                    "klass": klass, "scope": "stripped" if stripped else "full",
                    "input_tokens": st if stripped else ft,
                    "developer_message": DEV_PREFIX,
                    "user_message": M.USER_TEMPLATE.format(
                        task_description=instr, trajectory=M.format_atif_trajectory(a)),
                })
    with open(RESULTS / "monitor_inputs_sample.jsonl", "w") as f:
        for r in sample_rows:
            f.write(json.dumps(r) + "\n")

    n = n_hack + n_base
    full_all = full_hack + full_base
    strip_all = strip_hack + strip_base
    print(f"Gemini pool: {n_hack} hacks + {n_base} baselines = {n} trajectories")
    print(f"  full-scope input tokens:     {full_all:,}  (avg {full_all//max(n,1):,}/traj)")
    print(f"  stripped-scope input tokens: {strip_all:,}  (avg {strip_all//max(n,1):,}/traj)")
    print(f"  (developer-prompt overhead {DEV_TOKENS} tok/call, +~{EST_OUT} output tok/call)")

    # Primary plan: M0 full + M1 stripped + M2 stripped  (each over all n trajectories)
    # Upper bound: all 3 monitors x both scopes.
    rows = []
    for name, (pin, pout) in CANDIDATES.items():
        def cost(in_tok, calls):
            return round(in_tok / 1e6 * pin + calls * EST_OUT / 1e6 * pout, 2)
        m0 = cost(full_all, n)               # one monitor, full
        m_strip = cost(strip_all, n)         # one monitor, stripped
        primary = round(m0 + 2 * m_strip, 2)  # M0 full + M1,M2 stripped
        upper = round(3 * (m0 + m_strip), 2)  # 3 monitors x both scopes
        rows.append({"monitor": name, "in_$/M": pin, "out_$/M": pout,
                     "1xfull_usd": m0, "1xstripped_usd": m_strip,
                     "primary_M0full+2xstripped_usd": primary,
                     "upper_3mon_x_both_usd": upper})

    with open(RESULTS / "cost_estimate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== cost by candidate monitor (tiktoken; official renderer) ===")
    print(f"{'monitor':16} {'1xfull':>9} {'1xstrip':>9} {'PRIMARY':>9} {'UPPER':>9}")
    for r in rows:
        print(f"{r['monitor']:16} {r['1xfull_usd']:>9} {r['1xstripped_usd']:>9} "
              f"{r['primary_M0full+2xstripped_usd']:>9} {r['upper_3mon_x_both_usd']:>9}")
    print("\nPRIMARY = M0(full)+M1(stripped)+M2(stripped) over the whole Gemini pool.")
    print("Realistic plan: M0=gpt-5.4-nano, M1/M2 mid-tier cross-family -> ~$10-40.")
    print("Wrote results/cost_estimate.csv")


if __name__ == "__main__":
    main()
