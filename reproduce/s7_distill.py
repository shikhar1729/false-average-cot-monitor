"""Section 7: does the innocent narration style distill? A LoRA fine-tune of Qwen3-8B narrates
29 held-out trajectories whose actions are frozen; the held-out sonnet monitor scores each version."""
import numpy as np

from lib import binom_two_sided, check, jsonl, monitor, note, num, report, section, threshold
from s3_setup import SONNET


def main():
    by_source = {}
    for r in jsonl("distill/lora.jsonl"):
        by_source.setdefault(r["source"], {})[r["sample_id"]] = r["score"]
    honest, target, base, tuned = (by_source[k] for k in ("honest", "obf", "base_gen", "sft_gen"))
    ids = sorted(tuned)

    section("Section 7: mean suspicion on 29 held-out trajectories")
    check("held-out trajectories", "29", str(len(ids)))
    for label, src, paper in [("LoRA narration", tuned, "0.15"), ("honest original", honest, "0.88"),
                              ("obfuscated target", target, "0.12"), ("un-tuned base", base, "0.24")]:
        check(label, paper, num(np.mean([src[i] for i in ids])))

    section("Section 7: paired against the un-tuned base")
    lower = sum(tuned[i] < base[i] for i in ids)
    higher = sum(tuned[i] > base[i] for i in ids)
    check("lower / tied / higher than base", "18 / 6 / 5", f"{lower} / {len(ids) - lower - higher} / {higher}")
    check("sign test p", "0.01", num(binom_two_sided(lower, lower + higher)))
    tau = threshold(monitor(SONNET)[1])
    caught_tuned = {i for i in ids if tuned[i] >= tau}
    caught_base = {i for i in ids if base[i] >= tau}
    check("caught at the frozen-pool threshold: LoRA vs base", "4 of 29 vs 11 of 29",
          f"{len(caught_tuned)} of {len(ids)} vs {len(caught_base)} of {len(ids)}")
    only_base, only_tuned = len(caught_base - caught_tuned), len(caught_tuned - caught_base)
    check("paired (exact McNemar) p", "0.07", num(binom_two_sided(only_base, only_base + only_tuned)))
    note("the train/held-out split (65 and 16 tasks, 386 and 90 steps) is rebuilt from the probe inputs, which hold rewrite texts: generate/distill/prepare.py")


if __name__ == "__main__":
    main()
    report()
