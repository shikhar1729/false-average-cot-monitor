"""OUT_DIR side check (local, free): which paper-pool rows (results/internals_inputs_obf.jsonl) hit the pod's
16384-token cap (pod_probe.py MAXTOK), and does dropping them change the paper probe AUCs (8B L9, 32B L36,
in-dist task-CV and orig->obf cross-train)? A capped row loses its trajectory tail and the answer prompt,
so its last-token feature is a mid-trajectory token.

  .venv/bin/python scripts/probe32b_truncation_check.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]; sys.path[:0] = [str(p) for p in HERE.parents[0].iterdir() if p.is_dir()] + [str(ROOT / "monitoring"), str(ROOT / "repo" / "monitoring")]
from probe_controls import crosstrain_oof
from probe_detector import detector, oof_indist

OUT = ROOT / "results" / "appendix" / "OUT_DIR"
JSONL = ROOT / "results" / "internals_inputs_obf.jsonl"
CFG = {"8B": (HERE / "probe_feats.pt", 9), "32B": (HERE / "probe_feats32b.pt", 36)}
MAXTOK = 16384


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in open(JSONL) if l.strip()]
    tr = []
    for r in rows:
        ids = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                      add_generation_prompt=True, enable_thinking=False)
        if len(ids) >= MAXTOK:
            tr.append((r["sample_id"], r["variant"], len(ids)))
    res = {"capped_rows": tr}
    trs = {s for s, _, _ in tr}
    for tag, (f, L) in CFG.items():
        d = torch.load(f, map_location="cpu", weights_only=False)
        variant, task, y = np.asarray(d["variants"]), np.asarray(d["tasks"]), np.asarray(d["labels"]).astype(int)
        sids, p = [], 0
        for r in rows:
            if p < len(variant) and r["variant"] == variant[p] and r["task_id"] == task[p]:
                sids.append(r["sample_id"]); p += 1
        assert p == len(variant)
        keep = ~np.isin(np.array(sids), list(trs)); X = d["feats"][L]
        for name, k in (("all", np.ones(len(y), bool)), ("excl_capped", keep)):
            m = k & np.isin(variant, ["obf", "legit"])
            oof = oof_indist(X[m], y[m], task[m]); ct, _ = crosstrain_oof(X[k], y[k], variant[k], task[k]); ok = ~np.isnan(ct)
            res[f"{tag}_{name}"] = {"n_legit": int((variant[m] == "legit").sum()), "indist_auc": detector(oof, y[m])["auc"],
                                    "crosstrain_auc": detector(ct[ok], y[k][ok])["auc"]}
    json.dump(res, open(OUT / "truncation_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
