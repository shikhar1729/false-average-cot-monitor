"""Cross-train + length controls for the monitor-activation probe, run LOCALLY on saved features
(no GPU, no model weights). Answers:

  (i)   CROSS-TRAIN: train the probe on ORIGINAL hacks (+legit), test on OBFUSCATED hacks (+legit),
        with folds grouped by task so no task and no hack-in-any-form leaks across the split. If AUC
        stays high, the probe reads a hack representation that survives the rewrite, not the
        obfuscator's surface style.
  (iii) LENGTH: legit trajectories are longer than hacks, so we check the probe is not a length meter:
        length-only AUC, probe AUC on a length-matched band, and Spearman(probe score, seq length).

Also reproduces the within-variant obf-vs-legit AUC (must match probe_results*.json) as a pipeline check,
then augments that JSON in place with the control fields.

  python scripts/probe/probe_controls.py --feats scripts/probe_feats.pt   --tokenizer Qwen/Qwen3-8B  --results scripts/probe_results.json
  python scripts/probe/probe_controls.py --feats scripts/probe_feats32b.pt --tokenizer Qwen/Qwen3-32B --results scripts/probe_results32b.json
"""
import argparse, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE)); _S = os.path.dirname(HERE); sys.path[:0] = [os.path.join(_S, d) for d in os.listdir(_S) if os.path.isdir(os.path.join(_S, d))] + [os.path.join(ROOT, "monitoring"), os.path.join(ROOT, "repo", "monitoring")]
JSONL = os.path.join(ROOT, "results", "internals_inputs_obf.jsonl")
MAXTOK = 16384


def cv_auc(X, y, groups):
    """Out-of-fold task-split logistic-probe AUC (identical to pod_probe.py)."""
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof, roc_auc_score(y, oof)


def crosstrain_oof(X, is_hack, variant, task):
    """Train on {orig,legit}, test on {obf,legit}, folds grouped by task. Returns OOF preds and the
    boolean mask of scored (obf|legit) rows. Each row is a test row in exactly one fold (task-disjoint),
    so no legit and no task is shared between the orig-train and obf-test sides."""
    pred = np.full(len(is_hack), np.nan)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(task))))
    for tr_idx, te_idx in gkf.split(X, is_hack, groups=task):
        tr_tasks, te_tasks = set(task[tr_idx]), set(task[te_idx])
        trm = np.isin(variant, ["orig", "legit"]) & np.isin(task, list(tr_tasks))
        tem = np.isin(variant, ["obf", "legit"]) & np.isin(task, list(te_tasks))
        sc = StandardScaler().fit(X[trm])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[trm]), is_hack[trm])
        pred[tem] = clf.predict_proba(sc.transform(X[tem]))[:, 1]
    scored = np.isin(variant, ["obf", "legit"]) & ~np.isnan(pred)
    return pred, scored


def align_lengths(variants, tasks, tokenizer_id):
    """Recompute the capped token length the model saw for each feature row, by re-tokenizing the JSONL
    in file order and greedy-aligning to the (variant,task) sequence stored in the .pt (rows the pod
    skipped are dropped). Verifies the aligned sequence matches exactly."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    rows = [json.loads(l) for l in open(JSONL) if l.strip()]

    def seqlen(r):
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None)
        return min(len(ids), MAXTOK)

    lengths, skipped, p = [], [], 0
    for r in rows:
        if p < len(variants) and r["variant"] == variants[p] and r["task_id"] == tasks[p]:
            lengths.append(seqlen(r)); p += 1
        else:
            skipped.append((r["sample_id"], r["variant"]))
    assert p == len(variants), f"alignment consumed {p}/{len(variants)} feature rows"
    assert len(lengths) == len(variants)
    print(f"  aligned {len(lengths)} rows; skipped {len(skipped)}: {skipped}", flush=True)
    return np.array(lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--results", required=True, help="probe_results*.json to augment in place")
    args = ap.parse_args()

    d = torch.load(args.feats, map_location="cpu", weights_only=False)
    layers = sorted(d["feats"].keys())
    variant = np.asarray(d["variants"]); task = np.asarray(d["tasks"]); is_hack = np.asarray(d["labels"]).astype(int)
    lengths = align_lengths(variant, task, args.tokenizer)
    out = {}

    # --- pipeline check: within-variant obf-vs-legit (must match probe_results*.json) ---
    wv_by_layer, oof_best, best_layer, best_auc = {}, None, None, -1
    m_obf = np.isin(variant, ["obf", "legit"])
    for L in layers:
        X = d["feats"][L][m_obf]
        oof, auc = cv_auc(X, is_hack[m_obf], task[m_obf])
        wv_by_layer[L] = auc
        if auc > best_auc:
            best_auc, best_layer, oof_best = auc, L, (oof, m_obf)
    out["within_obf_auc_by_layer_recomputed"] = {str(k): v for k, v in wv_by_layer.items()}
    print(f"  within-variant obf AUC (recomputed): best={best_auc:.4f} @L{best_layer}", flush=True)

    # --- (i) cross-train: train orig(+legit) -> test obf(+legit), task-grouped ---
    ct_by_layer = {}
    for L in layers:
        pred, scored = crosstrain_oof(d["feats"][L], is_hack, variant, task)
        ct_by_layer[L] = float(roc_auc_score(is_hack[scored], pred[scored]))
    ct_best_layer = max(ct_by_layer, key=ct_by_layer.get)
    out["crosstrain_obf_auc_by_layer"] = {str(k): v for k, v in ct_by_layer.items()}
    out["crosstrain_obf_best_auc"] = ct_by_layer[ct_best_layer]
    out["crosstrain_obf_best_layer"] = int(ct_best_layer)
    out["crosstrain_obf_auc_at_within_best_layer"] = ct_by_layer[best_layer]
    print(f"  CROSS-TRAIN obf AUC (orig->obf, task-grouped): best={ct_by_layer[ct_best_layer]:.4f} @L{ct_best_layer} "
          f"| at within-best L{best_layer}={ct_by_layer[best_layer]:.4f}", flush=True)

    # --- (iii) length controls ---
    def length_auc(pos_variant):
        m = np.isin(variant, [pos_variant, "legit"])
        y = is_hack[m]; s = lengths[m].astype(float)
        a = roc_auc_score(y, s)                       # higher length -> hack?
        return max(a, 1 - a), a                        # discriminability, raw
    out["length_only_obf_auc"], out["length_only_obf_auc_raw"] = length_auc("obf")
    out["length_only_orig_auc"], out["length_only_orig_auc_raw"] = length_auc("orig")
    out["length_tok_median"] = {v: float(np.median(lengths[variant == v])) for v in ("obf", "orig", "legit")}

    # length-matched band: overlap of obf-hack and legit token lengths; recompute within-variant AUC there
    lo = max(lengths[(variant == "obf")].min(), lengths[variant == "legit"].min())
    hi = min(lengths[(variant == "obf")].max(), lengths[variant == "legit"].max())
    band = m_obf & (lengths >= lo) & (lengths <= hi)
    yb = is_hack[band]
    lm_by_layer = {}
    if len(np.unique(yb)) == 2:
        for L in layers:
            _, auc = cv_auc(d["feats"][L][band], yb, task[band])
            lm_by_layer[L] = auc
    lm_best_layer = max(lm_by_layer, key=lm_by_layer.get) if lm_by_layer else None
    out["length_matched_band_tok"] = [int(lo), int(hi)]
    out["length_matched_n"] = {"obf_hack": int((band & (variant == "obf")).sum()),
                               "legit": int((band & (variant == "legit")).sum())}
    a_band = roc_auc_score(yb, lengths[band].astype(float))
    out["length_only_obf_auc_in_band"] = max(a_band, 1 - a_band)
    out["length_matched_obf_auc_by_layer"] = {str(k): v for k, v in lm_by_layer.items()}
    out["length_matched_obf_best_auc"] = lm_by_layer[lm_best_layer] if lm_best_layer else None
    out["length_matched_obf_auc_at_within_best_layer"] = lm_by_layer.get(best_layer)

    # Spearman(probe OOF score, length) within each class at the within-best layer
    oof, mask = oof_best
    idx = np.where(mask)[0]
    sc_full = np.full(len(variant), np.nan); sc_full[idx] = oof
    out["probe_score_vs_length_spearman"] = {}
    for cls, vv in (("legit", "legit"), ("obf_hack", "obf")):
        sel = (variant == vv) & mask
        rho, _ = spearmanr(sc_full[sel], lengths[sel])
        out["probe_score_vs_length_spearman"][cls] = float(rho)
    print(f"  length-only AUC: obf={out['length_only_obf_auc']:.3f} orig={out['length_only_orig_auc']:.3f}"
          f" | length-matched obf AUC best={out['length_matched_obf_best_auc']}"
          f" (n obf={out['length_matched_n']['obf_hack']}/legit={out['length_matched_n']['legit']})"
          f" | Spearman(score,len) legit={out['probe_score_vs_length_spearman']['legit']:.2f}"
          f" obf={out['probe_score_vs_length_spearman']['obf_hack']:.2f}", flush=True)

    # --- persist: augment the existing results JSON in place (add keys, keep originals) ---
    res = json.load(open(args.results))
    res["controls"] = out
    json.dump(res, open(args.results, "w"), indent=2)
    print(f"  augmented {args.results} with res['controls']", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
