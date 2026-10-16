"""Cross-train and length controls for the 8B probe, run locally on the saved activations.

Reads runs/probe/probe_feats_8b.pt and internals_inputs_obf.jsonl and adds a "controls" block to
runs/probe/probe_results_8b.json (Appendix I). Cross-train: fit on original hacks and legit, test on
obfuscated hacks and legit, folds grouped by task. Length: token length alone, and the probe inside the
band where hack and legit lengths overlap. data/probe/probe_8b.json records four values, rounded, under
controls_recorded: crosstrain_obf_best_auc, length_only_obf_auc, length_matched_obf_best_auc, length_only_obf_auc_in_band.
Run: python -m generate.probe.controls_8b. Needs torch, transformers and the Qwen3-8B tokenizer; free.
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from generate.probe.linear_probe import crosstrain_oof, task_cv_scores
from generate.probe.pod import RUNS

FEATS = RUNS / "probe_feats_8b.pt"
RESULTS = RUNS / "probe_results_8b.json"
INPUTS = RUNS / "internals_inputs_obf.jsonl"
TOKENIZER = "Qwen/Qwen3-8B"
MAXTOK = 16384


def token_lengths(variants, tasks):
    """The capped token length the model saw for each feature row: re-tokenize the inputs in file order and
    walk them against the (variant, task) sequence in the features, dropping rows the pod skipped."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def seqlen(r):
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None)
        return min(len(ids), MAXTOK)

    lengths, skipped, p = [], [], 0
    for r in (json.loads(line) for line in open(INPUTS) if line.strip()):
        if p < len(variants) and r["variant"] == variants[p] and r["task_id"] == tasks[p]:
            lengths.append(seqlen(r))
            p += 1
        else:
            skipped.append((r["sample_id"], r["variant"]))
    assert p == len(variants), f"alignment consumed {p}/{len(variants)} feature rows"
    print(f"  aligned {len(lengths)} rows; skipped {len(skipped)}: {skipped}", flush=True)
    return np.array(lengths)


def within_variant(feats, layers, is_hack, variant, task):
    """Obfuscated hacks vs legit by layer; must match the pod's RESULTS block (a pipeline check)."""
    m = np.isin(variant, ["obf", "legit"])
    by_layer, best = {}, (-1, None, None)
    for L in layers:
        oof = task_cv_scores(feats[L][m], is_hack[m], task[m])
        by_layer[L] = roc_auc_score(is_hack[m], oof)
        if by_layer[L] > best[0]:
            best = (by_layer[L], L, oof)
    print(f"  within-variant obf AUC (recomputed): best={best[0]:.4f} @L{best[1]}", flush=True)
    return by_layer, best[1], best[2], m


def cross_train(feats, layers, is_hack, variant, task, within_best):
    by_layer = {}
    for L in layers:
        pred, scored = crosstrain_oof(feats[L], is_hack, variant, task)
        by_layer[L] = float(roc_auc_score(is_hack[scored], pred[scored]))
    best = max(by_layer, key=by_layer.get)
    print(f"  cross-train obf AUC (orig->obf, task-grouped): best={by_layer[best]:.4f} @L{best} "
          f"| at within-best L{within_best}={by_layer[within_best]:.4f}", flush=True)
    return {"crosstrain_obf_auc_by_layer": {str(k): v for k, v in by_layer.items()},
            "crosstrain_obf_best_auc": by_layer[best], "crosstrain_obf_best_layer": int(best),
            "crosstrain_obf_auc_at_within_best_layer": by_layer[within_best]}


def length_controls(feats, layers, is_hack, variant, task, lengths, m_obf, within_best):
    out = {}

    def length_auc(pos_variant):
        m = np.isin(variant, [pos_variant, "legit"])
        a = roc_auc_score(is_hack[m], lengths[m].astype(float))
        return max(a, 1 - a), a          # discriminability either way, and the raw AUC (longer = hack)
    out["length_only_obf_auc"], out["length_only_obf_auc_raw"] = length_auc("obf")
    out["length_only_orig_auc"], out["length_only_orig_auc_raw"] = length_auc("orig")
    out["length_tok_median"] = {v: float(np.median(lengths[variant == v])) for v in ("obf", "orig", "legit")}

    # the band where obfuscated-hack and legit lengths overlap; the probe refit inside it
    lo = max(lengths[variant == "obf"].min(), lengths[variant == "legit"].min())
    hi = min(lengths[variant == "obf"].max(), lengths[variant == "legit"].max())
    band = m_obf & (lengths >= lo) & (lengths <= hi)
    yb = is_hack[band]
    in_band = {}
    if len(np.unique(yb)) == 2:
        for L in layers:
            in_band[L] = roc_auc_score(yb, task_cv_scores(feats[L][band], yb, task[band]))
    best = max(in_band, key=in_band.get) if in_band else None
    a_band = roc_auc_score(yb, lengths[band].astype(float))
    out.update({"length_matched_band_tok": [int(lo), int(hi)],
                "length_matched_n": {"obf_hack": int((band & (variant == "obf")).sum()),
                                     "legit": int((band & (variant == "legit")).sum())},
                "length_only_obf_auc_in_band": max(a_band, 1 - a_band),
                "length_matched_obf_auc_by_layer": {str(k): v for k, v in in_band.items()},
                "length_matched_obf_best_auc": in_band[best] if best else None,
                "length_matched_obf_auc_at_within_best_layer": in_band.get(within_best)})
    return out


def score_length_spearman(variant, lengths, oof, mask):
    """Spearman between the probe's out-of-fold score (best within-variant layer) and length, per class."""
    score = np.full(len(variant), np.nan)
    score[np.where(mask)[0]] = oof
    out = {}
    for cls, v in (("legit", "legit"), ("obf_hack", "obf")):
        sel = (variant == v) & mask
        out[cls] = float(spearmanr(score[sel], lengths[sel])[0])
    return out


def main():
    d = torch.load(FEATS, map_location="cpu", weights_only=False)
    feats, layers = d["feats"], sorted(d["feats"].keys())
    variant, task = np.asarray(d["variants"]), np.asarray(d["tasks"])
    is_hack = np.asarray(d["labels"]).astype(int)
    lengths = token_lengths(variant, task)

    by_layer, best_layer, best_oof, m_obf = within_variant(feats, layers, is_hack, variant, task)
    out = {"within_obf_auc_by_layer_recomputed": {str(k): v for k, v in by_layer.items()}}
    out.update(cross_train(feats, layers, is_hack, variant, task, best_layer))
    out.update(length_controls(feats, layers, is_hack, variant, task, lengths, m_obf, best_layer))
    out["probe_score_vs_length_spearman"] = score_length_spearman(variant, lengths, best_oof, m_obf)
    print(f"  length-only AUC: obf={out['length_only_obf_auc']:.3f} orig={out['length_only_orig_auc']:.3f}"
          f" | length-matched obf AUC best={out['length_matched_obf_best_auc']}"
          f" (n obf={out['length_matched_n']['obf_hack']}/legit={out['length_matched_n']['legit']})", flush=True)

    res = json.load(open(RESULTS))
    res["controls"] = out
    json.dump(res, open(RESULTS, "w"), indent=2)
    print(f"added controls to {RESULTS}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
