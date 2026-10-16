"""Section 5 (first half), Table 2 and Appendix E: trace-only defenses, the reference-solution
defense, and the faithfulness of the rewrite."""
import numpy as np

from lib import caught, check, jsonl, monitor, num, pct, pearson, report, section, threshold, span
from s3_setup import SONNET

JUDGES = ["openai/gpt-5.4", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash", "qwen/qwen3-32b"]


def legit(name, key):
    return [r[key] for r in jsonl(f"defenses/legit_{name}.jsonl") if r[key] is not None]


def main():
    sonnet_hacks, sonnet_legit = monitor(SONNET)
    sonnet_full_legit = {r["sample_id"]: r["score"] for r in jsonl(SONNET) if not r["is_hack"]}
    tau = {"baseline": threshold(sonnet_legit), "skeptic": threshold(legit("skeptic", "skeptic")),
           "consistency": threshold(legit("consistency", "consistency")), "oracle": threshold(legit("oracle", "oracle")),
           "intentblind": threshold(legit("intent_blind", "intentblind")),
           "adaptive": threshold(legit("obfuscation_aware", "adaptive"))}

    section("Table 2: trace-only defenses on the 150-hack draw")
    trace = [r for r in jsonl("defenses/trace_only.jsonl") if r["obf_status"] == "ok"]
    headline = {r["sid"]: r for r in jsonl("attack/headline.jsonl")}
    check("baseline: before", "95%", pct(*caught([sonnet_hacks[r["sid"]] for r in trace], tau["baseline"])))
    check("baseline: after", "3%", pct(*caught([r["obf_baseline"] for r in trace], tau["baseline"])))
    check("skeptic: before", "94%", pct(*caught([headline[r["sid"]]["orig_skeptic"] for r in trace], tau["skeptic"])))
    check("skeptic: after", "10%", pct(*caught([r["obf_skeptic"] for r in trace], tau["skeptic"])))
    check("consistency: before", "95%", pct(*caught([r["orig_consistency"] for r in trace], tau["consistency"])))
    check("consistency: after", "5%", pct(*caught([r["obf_consistency"] for r in trace], tau["consistency"])))
    check("oracle-localized: after", "4%", pct(*caught([r["obf_oracle"] for r in trace], tau["oracle"])))

    section("Table 2: intent-blind monitor on all 251 subset hacks")
    blind = [r for r in jsonl("defenses/intent_blind.jsonl") if r["obf_status"] == "ok"]
    check("intent-blind: n", "251", str(len(blind)))
    check("intent-blind: before", "88%", pct(*caught([r["orig_intentblind"] for r in blind], tau["intentblind"])))
    check("intent-blind: after", "4%", pct(*caught([r["obf_intentblind"] for r in blind], tau["intentblind"])))
    check("intent-blind: mean suspicion, honest narration", "0.84", num(np.mean([r["orig_intentblind"] for r in blind])))
    check("intent-blind: mean suspicion, rewritten narration", "0.14", num(np.mean([r["obf_intentblind"] for r in blind])))
    evade_all = sum(all(r[f"obf_{k}"] < tau[k] for k in ("baseline", "skeptic", "consistency", "intentblind")) for r in blind)
    check("evade all four unprimed monitors at once", "87% (219 of 251)", f"{pct(evade_all, len(blind))} ({evade_all} of {len(blind)})")

    section("Table 2: obfuscation-aware monitor (primed with two examples), 100 hacks")
    aware = [r for r in jsonl("defenses/obfuscation_aware.jsonl") if r["obf_status"] == "ok"]
    check("obfuscation-aware: before", "89%", pct(*caught([r["orig_adaptive"] for r in aware], tau["adaptive"])))
    check("obfuscation-aware: after", "20%", pct(*caught([r["obf_adaptive"] for r in aware], tau["adaptive"])))

    section("Table 2: external ground truth (a genuine reference solution), 150 hacks")
    ref = jsonl("defenses/reference.jsonl")
    tau_ref = threshold([r["score"] for r in ref if r["kind"] == "neg"])
    k, n = caught([r["score"] for r in ref if r["kind"] == "hack" and r["obf_status"] == "ok"], tau_ref)
    check("reference solution: after", "48% (72 of 150)", f"{pct(k, n)} ({k} of {n})")

    section("Section 5 and Appendix E: the rewrite stays action-faithful")
    coh = jsonl("faithfulness/coherence.jsonl")
    check("faithfulness judge: rewrite", "0.96", num(np.mean([r["faith_obf"] for r in coh])))
    check("faithfulness judge: honest original", "0.97", num(np.mean([r["faith_orig"] for r in coh])))
    four = [r for r in jsonl("faithfulness/four_judges.jsonl") if r["status"] == "ok"]
    mean = lambda j, k, v: np.mean([r[f"{j}|{k}|{v}"] for r in four if r.get(f"{j}|{k}|{v}") is not None])
    check("four judges: rewrite faithfulness range", "0.94 to 0.97",
          span(min(mean(j, 'faith', 'obf') for j in JUDGES), max(mean(j, 'faith', 'obf') for j in JUDGES), 2, "", 1))
    check("four judges: original faithfulness range", "0.94 to 0.98",
          span(min(mean(j, 'faith', 'orig') for j in JUDGES), max(mean(j, 'faith', 'orig') for j in JUDGES), 2, "", 1))
    check("four judges: contradiction rate, rewrite", "3 to 28%",
          span(min(mean(j, 'contra', 'obf') for j in JUDGES), max(mean(j, 'contra', 'obf') for j in JUDGES)))
    check("four judges: contradiction rate, original", "0 to 35%",
          span(min(mean(j, 'contra', 'orig') for j in JUDGES), max(mean(j, 'contra', 'orig') for j in JUDGES)))

    section("Appendix E: rewrites forced to be unfaithful (adversarial fabrication), 50 hacks")
    fab = jsonl("faithfulness/fabrication.jsonl")
    gpt = np.array([r["faith"]["gpt"] for r in fab])
    gem = np.array([r["faith"]["gemini"] for r in fab])
    mon = np.array([r["monitor"] for r in fab])
    check("gpt judge: mean faithfulness", "0.35", num(gpt.mean()))
    check("gemini judge: mean faithfulness", "0.64", num(gem.mean()))
    check("faithfulness vs suspicion, gpt judge (Pearson r)", "-0.44", num(pearson(gpt, mon)))
    check("faithfulness vs suspicion, gemini judge (Pearson r)", "-0.19", num(pearson(gem, mon)))
    rates = []
    for label, mask, paper in [("all fabricated rewrites", np.ones(len(fab), bool), "68% of 50"),
                               ("gpt faithfulness < 0.5", gpt < 0.5, "60% of 40"), ("gpt faithfulness < 0.3", gpt < 0.3, "48% of 25"),
                               ("gemini faithfulness < 0.5", gem < 0.5, "45% of 11"), ("both judges < 0.5", (gpt < 0.5) & (gem < 0.5), "40% of 10")]:
        sel = mon[mask]
        rates.append((sel < tau["baseline"]).mean())
        check(f"evade at the nominal 5% threshold, {label}", paper, f"{pct((sel < tau['baseline']).sum(), len(sel))} of {len(sel)}")
    check("Section 5: range of those evasion rates", "40 to 68%", span(min(rates), max(rates)))

    section("Appendix B: every defense on the shared 100-hack core, thresholds from the same 150 legit solves")
    core = {r["sid"] for r in aware}
    same150 = [sonnet_full_legit[i] for i in [r["sid"] for r in jsonl("defenses/legit_skeptic.jsonl")]]
    tr = {r["sid"]: r for r in trace}
    ib = {r["sid"]: r for r in blind}
    oa = {r["sid"]: r for r in aware}
    for label, legit_scores, hack_scores, paper in [
            ("baseline", same150, [tr[s]["obf_baseline"] for s in core], "1%"),
            ("skeptic", legit("skeptic", "skeptic"), [tr[s]["obf_skeptic"] for s in core], "11%"),
            ("consistency", legit("consistency", "consistency"), [tr[s]["obf_consistency"] for s in core], "6%"),
            ("oracle-localized", legit("oracle", "oracle"), [tr[s]["obf_oracle"] for s in core], "6%"),
            ("intent-blind", legit("intent_blind", "intentblind"), [ib[s]["obf_intentblind"] for s in core], "5%"),
            ("obfuscation-aware", legit("obfuscation_aware", "adaptive"), [oa[s]["obf_adaptive"] for s in core], "20%"),
            ("reference solution", [r["score"] for r in ref if r["kind"] == "neg"],
             [r["score"] for r in ref if r["kind"] == "hack" and r["sid"] in core], "50%")]:
        check(f"{label}: catch after the rewrite", paper, pct(*caught(hack_scores, threshold(legit_scores))))


if __name__ == "__main__":
    main()
    report()
