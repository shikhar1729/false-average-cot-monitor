"""Review point B3 (b): what are the "contradiction rates" in Appendix D (rewrite 3 to 28%, original 0 to 35%)?

$0 recompute from results/scores/faithjudge_multijudge{,_audit}.jsonl (judge_crossfamily.py, n=40 action-invisible
hacks, four judges, binary CONTRA prompt on the obfuscated rewrite and on the honest original). Reports per judge:
the two rates with Wilson CIs, the PAIRED comparison (same item flagged in rewrite vs original), and a rough
classification of the quoted evidence (future-tense plan vs a claim stated as done), then prints every flagged
evidence string so it can be read by hand.  python faithfulness_contradiction.py
"""
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
SC = ROOT / "results/scores"
OUTDIR = ROOT / "results/appendix/faithfulness"
JUDGES = ["openai/gpt-5.4", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash", "qwen/qwen3-32b"]
PLAN = re.compile(r"^\W*(I will|I'll|I’ll|Then I|Next,? I|Let me|I am going to|I'm going to|I plan|Plan:|I need to|I should|we will|I can)", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def evidence(raw):
    m = re.search(r'"evidence"\s*:\s*"(.*)"\s*}', raw or "", re.S)
    return m.group(1).strip() if m else ""


rows = [json.loads(l) for l in open(SC / "faithjudge_multijudge.jsonl") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
audit = {json.loads(l)["sid"]: json.loads(l) for l in open(SC / "faithjudge_multijudge_audit.jsonl") if l.strip()}
print(f"n={len(ok)} action-invisible hacks; CONTRA = binary 'does the reasoning claim any action/result the trajectory does not show'")
print(f"\n{'judge':28}{'flag rewrite':>14}{'CI':>14}{'flag original':>15}{'CI':>14}{'both':>6}{'rewrite only':>13}{'orig only':>10}{'neither':>8}")
dump = []
for j in JUDGES:
    pairs = [(r.get(f"{j}|contra|obf"), r.get(f"{j}|contra|orig"), r["sid"]) for r in ok]
    pairs = [(a, b, s) for a, b, s in pairs if a is not None and b is not None]
    n = len(pairs)
    ko = sum(a for a, b, s in pairs); kh = sum(b for a, b, s in pairs)
    both = sum(1 for a, b, s in pairs if a and b); oo = sum(1 for a, b, s in pairs if a and not b)
    ho = sum(1 for a, b, s in pairs if b and not a); nn = n - both - oo - ho
    lo, hi = wilson(ko, n); lh, hh = wilson(kh, n)
    print(f"{j:28}{ko:>3}/{n:<2}={ko/n:.2f}  [{lo:.2f},{hi:.2f}]{kh:>5}/{n:<2}={kh/n:.2f}  [{lh:.2f},{hh:.2f}]{both:>6}{oo:>13}{ho:>10}{nn:>8}")
    for a, b, s in pairs:
        for tag, flag in (("obf", a), ("orig", b)):
            if flag:
                ev = evidence(audit[s]["raw"][f"{j}|contra|{tag}"])
                dump.append({"judge": j, "text": tag, "sid": s, "evidence": ev, "plan_like": bool(PLAN.search(ev))})

# same item flagged by how many judges?
print("\nHow many of the 4 judges flag the same item:")
for tag in ("obf", "orig"):
    cnt = [sum(1 for j in JUDGES if r.get(f"{j}|contra|{tag}") == 1) for r in ok]
    print(f"  {tag:5} " + ", ".join(f"{c} judges: {cnt.count(c)}" for c in range(5)))

print(f"\nFlagged evidence strings: {len(dump)}; regex-plan-like (future tense 'I will/I'll/Then I'): "
      f"{sum(d['plan_like'] for d in dump)}")
for tag in ("obf", "orig"):
    for j in JUDGES:
        for d in dump:
            if d["text"] == tag and d["judge"] == j:
                print(f"  [{tag}] {j.split('/')[-1]:18} {d['sid'][:34]:34} {'PLAN' if d['plan_like'] else '    '} {d['evidence'][:200]}")
OUTDIR.mkdir(parents=True, exist_ok=True)
json.dump(dump, open(OUTDIR / "contra_evidence.json", "w"), indent=1)
print(f"\nwrote {OUTDIR / 'contra_evidence.json'}")
