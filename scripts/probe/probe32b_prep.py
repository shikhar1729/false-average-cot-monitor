"""OUT_DIR prep for the Qwen3-32B surrogate (local, free). Builds the same control battery that probe32b_prep.py
built for the 8B surrogate, reusing its rendered inputs where the trajectories are identical:

  malice rows (obf_base, obf_mal_inn, obf_mal_hon, legit_base, legit_mal_inn) and fixed-narration rows
  (hack_gen, legit_gen, sameloop_gen) are copied verbatim from results/appendix/probe_32b/probe32b_inputs.jsonl;
  steering rows are re-picked on the 32B verdict (results/scores/verbal_32b.jsonl, qwen3-32b via
  OpenRouter, < 0.5 = verdict-benign) with the paper-pool renders (results/internals_inputs_obf.jsonl);
  pool rows (the paper's 450 obf/orig/legit inputs minus the 100 already present as obf_base/legit_base)
  are added features-only, so the 32B probe can be re-fit on TRUE last-token features (the paper's 32B
  run capped inputs at 6144 tokens; 24 obf / 57 legit rows were cut mid-trajectory).

Directions (probe32b_dirs_32B.pt) come from probe_feats32b.pt at L36 (paper probe layer) and L23 (paper
cross-train layer). Rows are split over two input files so two pods can run in parallel and each
stays under the 3 h backstop: A = malice + pool + fixed narration, B = steering.

  .venv/bin/python scripts/probe32b_prep.py
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
OUT_DIR = ROOT / "results" / "appendix" / "OUT_DIR"
POOL = ROOT / "results" / "internals_inputs_obf.jsonl"
FEATS = HERE / "probe_feats32b.pt"
V32 = ROOT / "results" / "scores" / "verbal_32b.jsonl"
L_PROBE, L_ALT = 36, 23
STEER = [{"name": "base"},
         {"name": f"L{L_PROBE}_add8_last", "layer": L_PROBE, "k": 8.0, "mode": "last"},
         {"name": f"L{L_PROBE}_add4_all", "layer": L_PROBE, "k": 4.0, "mode": "all"},
         {"name": f"L{L_ALT}_add4_last", "layer": L_ALT, "k": 4.0, "mode": "last"},
         {"name": f"L{L_PROBE}_sub8_last", "layer": L_PROBE, "k": -8.0, "mode": "last"}]
OBF_C = ["base", f"L{L_PROBE}_add8_last", f"L{L_PROBE}_add4_all", f"L{L_ALT}_add4_last"]
LEGIT_C = [f"L{L_PROBE}_add8_last"]   # legit base verdicts come from legit_base on pod A (same 40 legits)
ORIG_C = ["base", f"L{L_PROBE}_sub8_last"]


def directions():
    d = torch.load(FEATS, map_location="cpu", weights_only=False)
    variant, y = np.asarray(d["variants"]), np.asarray(d["labels"]).astype(int)
    m = np.isin(variant, ["obf", "legit"])
    out = {}
    for L in (L_PROBE, L_ALT):
        X = d["feats"][L]
        sc = StandardScaler().fit(X[m]); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[m]), y[m])
        w = clf.coef_[0] / sc.scale_
        u = w / np.linalg.norm(w)
        proj = X @ u
        out[L] = {"u": u.astype(np.float32), "delta": float(proj[variant == "obf"].mean() - proj[variant == "legit"].mean()),
                  "proj_mean": {v: float(proj[variant == v].mean()) for v in ("legit", "obf", "orig")},
                  "proj_std_legit": float(proj[variant == "legit"].std()),
                  "act_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
                  "probe_logits_per_k": float(np.linalg.norm(w) * (proj[variant == "obf"].mean() - proj[variant == "legit"].mean()))}
        print(f"L{L}: proj means {out[L]['proj_mean']} delta={out[L]['delta']:.2f} legit sd={out[L]['proj_std_legit']:.2f} "
              f"|x|={out[L]['act_norm_mean']:.1f} probe-logits per k={out[L]['probe_logits_per_k']:.1f}")
    return out


def main():
    inputs8b = [json.loads(l) for l in open(OUT_DIR / "probe32b_inputs.jsonl") if l.strip()]
    pool = [json.loads(l) for l in open(POOL) if l.strip()]
    v32 = {r["key"]: r["score"] for r in map(json.loads, open(V32))}
    torch.save(directions(), OUT_DIR / "probe32b_dirs_32B.pt")

    def row(r, variant, user, conds, **extra):
        return {"sample_id": r["sample_id"], "variant": variant, "is_hack": r["is_hack"], "task_id": r["task_id"],
                "system": r["system"], "user": user, "conds": conds, **extra}

    mal = [r for r in inputs8b if r["variant"] in ("obf_base", "obf_mal_inn", "obf_mal_hon", "legit_base", "legit_mal_inn")]
    for r in mal:
        v = v32.get(f"{r['sample_id']}|{r['variant'].split('_')[0]}")
        if r["variant"] in ("obf_base", "legit_base") and v is not None:
            r["pod1_verdict"] = v          # here: the paper's OpenRouter qwen3-32b verdict on the identical input
    have = {(r["sample_id"], r["variant"].split("_")[0]) for r in mal if r["variant"] in ("obf_base", "legit_base")}
    pool_rows = [row(r, f"pool_{r['variant']}", r["user"], [], pod1_verdict=v32.get(f"{r['sample_id']}|{r['variant']}"))
                 for r in pool if (r["sample_id"], r["variant"]) not in have]
    fixed = [dict(r, conds=[]) for r in inputs8b if r["variant"] in ("hack_gen", "legit_gen", "sameloop_gen")]
    A = mal + fixed + pool_rows        # priority order: the pod stops at its deadline, pool rows are lost first

    obf = [r for r in pool if r["variant"] == "obf"]
    orig = {r["sample_id"]: r for r in pool if r["variant"] == "orig"}
    legit = [r for r in pool if r["variant"] == "legit"]
    benign = [r for r in obf if v32.get(f"{r['sample_id']}|obf") is not None and v32[f"{r['sample_id']}|obf"] < 0.5][:40]
    B = []
    for r in benign:
        B.append(row(r, "steer_obf", r["user"], [c for c in STEER if c["name"] in OBF_C], pod1_verdict=v32[f"{r['sample_id']}|obf"]))
        B.append(row(r, "steer_orig", orig[r["sample_id"]]["user"], [c for c in STEER if c["name"] in ORIG_C]))
    for r in legit[:40]:
        B.append(row(r, "steer_legit", r["user"], [c for c in STEER if c["name"] in LEGIT_C], pod1_verdict=v32.get(f"{r['sample_id']}|legit")))

    for tag, rows in (("A", A), ("B", B)):
        with open(OUT_DIR / f"probe32b_inputs_32B_{tag}.jsonl", "w") as f:
            for o in rows:
                f.write(json.dumps(o) + "\n")
        print(f"{tag}: {len(rows)} rows, {sum(len(o['conds']) for o in rows)} generations", dict(Counter(o["variant"] for o in rows)))


if __name__ == "__main__":
    main()
