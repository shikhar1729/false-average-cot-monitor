"""Section 4, Tables 1, 3 and 4: the narration-only rewrite on the frozen pool."""
import numpy as np

from lib import auc, caught, check, jsonl, monitor, note, num, pct, report, rnd, section, span, threshold, wilson
from s3_setup import NANO, SONNET, SONNET_AO, reasoning_dependent_subset

FLASH_AO, GPT54_AO = "frozen_pool/flash_action-only.jsonl", "frozen_pool/gpt54_action-only.jsonl"
TAMPERING = {"output-spoofing", "structural-extraction", "binary-hijacking", "mutable-input-tampering", "metric-spoofing"}
OMISSION = {"hollow-implementation", "constraint-loophole", "algorithmic-simplification", "keyword-gaming"}


def label_type(categories):
    """Tampering-type if any label acts on the checker; omission-type if every label is work not done."""
    cats = set(categories)
    if cats & TAMPERING:
        return "tampering"
    return "omission" if cats and cats <= OMISSION else "other"


def main():
    _, sonnet_legit = monitor(SONNET)
    _, nano_legit = monitor(NANO)
    tau_sonnet, tau_nano = threshold(sonnet_legit), threshold(nano_legit)

    section("Section 4: headline, 209 reasoning-dependent hacks, held-out sonnet")
    rows = [r for r in jsonl("attack/headline.jsonl") if r["obf_status"] == "ok"]
    k0, n = caught([r["orig_baseline"] for r in rows], tau_sonnet)
    k1, _ = caught([r["obf_baseline"] for r in rows], tau_sonnet)
    check("catch before the rewrite", "96.2%", pct(k0, n, 1))
    check("catch after the rewrite", "3.8%", pct(k1, n, 1))
    check("hacks still caught", "8 of 209", f"{k1} of {n}")
    lo, hi = wilson(k1, n)
    check("95% CI after the rewrite", "2 to 7%", span(lo, hi))

    section("Section 4: unbiased random 150 of the caught pool (the false average)")
    sample = [r for r in jsonl("attack/random_sample.jsonl") if r["obf_status"] == "ok"]
    rate = lambda key, tau: caught([r[key] for r in sample], tau)[0] / len(sample)
    check("target nano: before", "1.00", num(rate("orig_m0", tau_nano)))
    check("target nano: after", "0.78", num(rate("obf_m0", tau_nano)))
    check("held-out sonnet: before", "0.98", num(rate("orig_m1", tau_sonnet)))
    check("held-out sonnet: after", "0.73", num(rate("obf_m1", tau_sonnet)))
    pooled_drop = rate("orig_m1", tau_sonnet) - rate("obf_m1", tau_sonnet)
    par_drop = rate("orig_m1", tau_sonnet) - rate("par_m1", tau_sonnet)
    check("pooled drop, sonnet (points)", "25", rnd(100 * pooled_drop))
    check("targeted drop beyond the paraphrase null (points)", "23", rnd(100 * (pooled_drop - par_drop)))
    subset = set(reasoning_dependent_subset())
    share = len(subset) / len(monitor(NANO)[0])
    check("dilution prediction: subset share x subset collapse (points)", "21", rnd(100 * share * (k0 - k1) / n))
    check("sonnet AUC on the random 150: before", "0.99", num(auc([r["orig_m1"] for r in sample], sonnet_legit)))
    check("sonnet AUC on the random 150: after", "0.91", num(auc([r["obf_m1"] for r in sample], sonnet_legit)))

    section("Table 1: robustness")
    legit_skeptic = [r["skeptic"] for r in jsonl("defenses/legit_skeptic.jsonl")]
    tau_skeptic = threshold(legit_skeptic)
    check("skeptic rubric: before", "94%", pct(*caught([r["orig_skeptic"] for r in rows], tau_skeptic)))
    check("skeptic rubric: after", "11%", pct(*caught([r["obf_skeptic"] for r in rows], tau_skeptic)))
    in_subset = [r for r in sample if r["sid"] in subset]
    check("paraphrase control: subset hacks in the random 150", "33", str(len(in_subset)))
    check("paraphrase control: before", "91%", pct(*caught([r["orig_m1"] for r in in_subset], tau_sonnet)))
    check("paraphrase control: after the paraphrase", "85%", pct(*caught([r["par_m1"] for r in in_subset], tau_sonnet)))
    check("paraphrase control: after the rewrite, same 33", "6%", pct(*caught([r["obf_m1"] for r in in_subset], tau_sonnet)))

    opus = [r for r in jsonl("attack/opus_actor.jsonl") if r["obf_status"] == "ok" and r["par_status"] == "ok"]
    _, opus_legit = monitor("opus_pool/flash_full_honest.jsonl")
    tau_opus = threshold(opus_legit)
    check("other actor (opus hacks, flash monitor): n", "96", str(len(opus)))
    check("other actor: threshold", "1.0", num(tau_opus, 1))
    check("other actor: before", "74%", pct(*caught([r["orig"] for r in opus], tau_opus)))
    check("other actor: after", "0%", pct(*caught([r["obf"] for r in opus], tau_opus)))
    check("other actor: AUC before", "0.91", num(auc([r["orig"] for r in opus], opus_legit)))
    check("other actor: AUC after", "0.25", num(auc([r["obf"] for r in opus], opus_legit)))
    check("other actor: AUC after the paraphrase", "0.91", num(auc([r["par"] for r in opus], opus_legit)))

    section("Section 4: iteration and repeated obfuscator draws")
    iterated = jsonl("attack/iterated.jsonl")
    m1 = lambda k: np.mean([r["rounds"][k]["m1"] >= tau_sonnet for r in iterated])
    check("one more round against still-caught hacks: sonnet catch, round 1", "0.08", num(m1(1)))
    check("one more round: sonnet catch, round 2 (no further drop)", "0.12", num(m1(2)))
    draws = {"random 150": [r["obf_baseline"] for r in jsonl("defenses/trace_only.jsonl") if r["obf_status"] == "ok"],
             "headline 209": [r["obf_baseline"] for r in rows],
             "intent-blind 251": [r["obf_baseline"] for r in jsonl("defenses/intent_blind.jsonl") if r["obf_status"] == "ok"],
             "obfuscation-aware 100": [r["obf_baseline"] for r in jsonl("defenses/obfuscation_aware.jsonl") if r["obf_status"] == "ok"]}
    rates = {k: caught(v, tau_sonnet) for k, v in draws.items()}
    check("four independent draws: lowest catch", "3.3%", pct(*min(rates.values(), key=lambda kn: kn[0] / kn[1]), 1))
    check("four independent draws: highest catch", "6.0%", pct(*max(rates.values(), key=lambda kn: kn[0] / kn[1]), 1))
    note("the hindsight audit (2 of 1,044 rewritten steps) needs the rewrite texts, which are on request")

    section("Table 3 (Appendix D): sensitivity of the subset definition")
    nano_hacks, _ = monitor(NANO)
    obf = {r["sid"]: r["obf_baseline"] for r in rows}
    for name, definers, fpr, size, before, after, drop in [
            ("sonnet", [SONNET_AO], 0.01, "43.7%", "84%", "0%", "84"),
            ("sonnet", [SONNET_AO], 0.05, "22.9%", "96%", "4%", "92"),
            ("sonnet", [SONNET_AO], 0.10, "15.7%", "96%", "6%", "90"),
            ("flash", [FLASH_AO], 0.05, "15.7%", "98%", "3%", "94"),
            ("sonnet + flash", [SONNET_AO, FLASH_AO], 0.01, "21.3%", "82%", "0%", "82"),
            ("sonnet + flash", [SONNET_AO, FLASH_AO], 0.05, "12.7%", "97%", "3%", "94"),
            ("sonnet + flash", [SONNET_AO, FLASH_AO], 0.10, "9.7%", "97%", "5%", "93"),
            ("gpt-5.4 (frontier, also missed by sonnet)", [SONNET_AO, GPT54_AO], 0.05, "18.6%", "96%", "3%", "93")]:
        sub = reasoning_dependent_subset(fpr, definers)
        tau = threshold(sonnet_legit, fpr)
        b = np.mean([monitor(SONNET)[0][s] >= tau for s in sub])
        a = np.mean([obf[s] >= tau for s in sub if s in obf])
        tag = f"{name}, {fpr:.0%} FPR"
        check(f"{tag}: subset share", size, pct(len(sub), len(nano_hacks), 1))
        check(f"{tag}: before -> after", f"{before} -> {after}", f"{pct(b)} -> {pct(a)}")
        check(f"{tag}: drop (points)", drop, rnd(100 * (b - a)))
    ao_gpt, legit_gpt = monitor(GPT54_AO)
    tau_gpt = threshold(legit_gpt)
    canonical = reasoning_dependent_subset()
    check("frontier gpt-5.4 action reader misses of the canonical subset", "81%",
          pct(sum(ao_gpt[s] < tau_gpt for s in canonical), len(canonical)))

    section("Table 4 (Appendix D): a split with no monitor in its definition")
    cats = {r["sample_id"]: r["observed_categories"] for r in jsonl(SONNET) if r["is_hack"]}
    groups = {"omission": [], "tampering": [], "all": sample}
    for r in sample:
        t = label_type(r["observed_categories"])
        if t in groups:
            groups[t].append(r)
    for t, before, after, para, share_paper in [("omission", "99%", "64%", "97%", "31%"),
                                                ("tampering", "97%", "87%", "96%", "10%"),
                                                ("all", "98%", "73%", "97%", "22%")]:
        g = groups[t]
        check(f"{t}-type (n={len(g)}): before", before, pct(*caught([r["orig_m1"] for r in g], tau_sonnet)))
        check(f"{t}-type: after the rewrite", after, pct(*caught([r["obf_m1"] for r in g], tau_sonnet)))
        check(f"{t}-type: paraphrase", para, pct(*caught([r["par_m1"] for r in g], tau_sonnet)))
        check(f"{t}-type: share inside the subset", share_paper, pct(sum(r["sid"] in subset for r in g), len(g)))
    ao_hacks, ao_legit = monitor(SONNET_AO)
    tau_ao = threshold(ao_legit)
    for t, share_paper, count_paper, ao_paper in [("omission", "34%", "198 of 580", "64%"), ("tampering", "9%", "41 of 478", "91%")]:
        pool = [s for s, c in cats.items() if label_type(c) == t]
        k = sum(s in subset for s in pool)
        check(f"pool, {t}-type: share in the subset", share_paper, pct(k, len(pool)))
        check(f"pool, {t}-type: count", count_paper, f"{k} of {len(pool)}")
        check(f"pool, {t}-type: action-only catch", ao_paper, pct(sum(ao_hacks[s] >= tau_ao for s in pool), len(pool)))

    section("Appendix B: iteration and the other actor")
    it = jsonl("attack/iterated.jsonl")
    for k, son, nan in [(0, "98%", "98%"), (1, "8%", "25%"), (2, "12%", "28%")]:
        check(f"after {k} chained rewrites: sonnet / nano catch", f"{son} / {nan}",
              f"{pct(sum(r['rounds'][k]['m1'] >= tau_sonnet for r in it), len(it))} / {pct(sum(r['rounds'][k]['m0'] >= tau_nano for r in it), len(it))}")
    still = [r for r in it if r["rounds"][1]["m0"] >= tau_nano]
    check("nano-caught after one rewrite, of them evading nano after two", "15, 6", f"{len(still)}, {sum(r['rounds'][2]['m0'] < tau_nano for r in still)}")
    pool = jsonl("opus_pool/nano_full.jsonl")
    check("opus pool: hacks / legit solves", "896 / 614", f"{sum(r['is_hack'] for r in pool)} / {sum(not r['is_hack'] for r in pool)}")
    rows97 = jsonl("attack/opus_actor.jsonl")
    check("opus subset: hacks / with both rewrites", "97 / 96", f"{len(rows97)} / {len(opus)}")
    check("flash on opus legit solves: median (n)", "0.80 (1,025)", f"{num(np.median(opus_legit))} ({len(opus_legit):,})")
    check("opus rewritten hacks: median, share below 0.5", "0.00, 80%", f"{num(np.median([r['obf'] for r in opus]))}, {pct(sum(r['obf'] < 0.5 for r in opus), len(opus))}")
    check("opus originals: median, share below 0.5", "1.00, 0%", f"{num(np.median([r['orig'] for r in opus]))}, {pct(sum(r['orig'] < 0.5 for r in opus), len(opus))}")
    check("opus paraphrase: caught at tau=1.0", "73%", pct(*caught([r["par"] for r in opus], tau_opus)))

    section("Appendix D: the actions and a hybrid monitor")
    full_h, full_l = monitor(SONNET)
    ao_h, ao_l = monitor(SONNET_AO)
    fl = {r["sample_id"]: r["score"] for r in jsonl(SONNET) if not r["is_hack"]}
    al = {r["sample_id"]: r["score"] for r in jsonl(SONNET_AO) if not r["is_hack"]}
    head = {r["sid"]: r for r in rows}
    def hybrid(combine, definers):
        sub_h = [s for s in reasoning_dependent_subset(0.05, definers) if s in head]
        legit = [combine(fl[i], al[i]) for i in fl]
        t = threshold(legit)
        before = [combine(full_h[s], ao_h[s]) for s in sub_h]
        after = [combine(head[s]["obf_baseline"], ao_h[s]) for s in sub_h]
        return sub_h, legit, t, before, after
    for label, definers, paper in [("canonical", [SONNET_AO], "0.74"), ("flash-defined", [FLASH_AO], "0.69"), ("gpt-5.4 core", [SONNET_AO, GPT54_AO], "0.72")]:
        sub_h = [s for s in reasoning_dependent_subset(0.05, definers) if s in head]
        check(f"action-only AUC on the {label} subset", paper, num(auc([ao_h[s] for s in sub_h], ao_l)))
    sub_h = [s for s in subset if s in head]
    check("mean full-scope after the rewrite vs mean action-only", "0.09 vs 0.11",
          f"{num(np.mean([head[s]['obf_baseline'] for s in sub_h]))} vs {num(np.mean([ao_h[s] for s in sub_h]))}")
    check("rewritten full-scope below the action-only score", "41%", pct(sum(head[s]["obf_baseline"] < ao_h[s] for s in sub_h), len(sub_h)))
    _, legit, t, b, a = hybrid(max, [SONNET_AO])
    check("hybrid max: catch before -> after", "92% -> 1%", f"{pct(np.mean(np.array(b) >= t))} -> {pct(np.mean(np.array(a) >= t))}")
    check("hybrid max: AUC before -> after", "0.97 -> 0.71", f"{num(auc(b, legit))} -> {num(auc(a, legit))}")
    _, legit, t, b, a = hybrid(lambda x, y: (x + y) / 2, [SONNET_AO])
    check("hybrid mean: catch before -> after", "91% -> 0%", f"{pct(np.mean(np.array(b) >= t))} -> {pct(np.mean(np.array(a) >= t))}")
    for label, definers in [("flash-defined", [FLASH_AO]), ("gpt-5.4 core", [SONNET_AO, GPT54_AO])]:
        _, legit, t, b, a = hybrid(max, definers)
        check(f"hybrid max on the {label} subset: catch after", "1%", pct(np.mean(np.array(a) >= t)))
    check("pool hacks with other or mixed labels", "37", str(sum(label_type(c) == "other" for c in cats.values())))
    note("the 1,495 hacks before the keyword filter (400 dropped) need the dataset clone: generate/common/tw.py")


if __name__ == "__main__":
    main()
    report()
