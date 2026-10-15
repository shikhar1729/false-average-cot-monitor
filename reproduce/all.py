"""Recompute every number in the paper from data/ and print it beside the paper's value.
Free: no API key, no GPU, numpy only. Exits non-zero if any number disagrees.

    python reproduce/all.py
"""
import lib
import s3_setup
import s4_attack
import s5_defenses
import s5_live
import s6_probe
import s7_distill

for module in (s3_setup, s4_attack, s5_defenses, s5_live, s6_probe, s7_distill):
    module.main()
lib.report()
