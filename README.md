# A False Average

Code and score files for *A False Average: Pooled CoT-Monitor Accuracy Conceals a Reasoning-Dependent Fragility*. The score files in `results/` are the inputs to every table; each `--analyze` (or default) mode below recomputes its numbers from them at no cost and needs no API key. Regenerating the scores needs an OpenRouter key and, for the live runs, probe captures and fine-tune, a rented GPU pod. The 32B probe analyses also need the captured activations, which are not included (large tensors).

```
scripts/common/         shared: tw_lib (parsing), run_monitor (drives the dataset monitor), monitor_rubrics, api_client, analysis_common
scripts/subset/         Sec 3, App B, App C: the reasoning-dependent subset, sensitivity, label split, Table 2 reproduction
scripts/attack/         Sec 4, Table 1: the rewrite, iteration, skeptic rubric, other actor family
scripts/defenses/       Table 2: skeptic, consistency, intent-blind, obfuscation-aware, reference-solution defenses
scripts/faithfulness/   App D: faithfulness judges, adversarial fabrication
scripts/live/           Sec 5, App G, App I: the live-agent sandbox runner and its launcher
scripts/probe/          Sec 6, App H: probe captures, 8B and 32B analyses, behavioral controls
scripts/sft/            Sec 6.3: distillation
monitoring/, prompts/   the Terminal Wrench authors' monitor.py, its rubric, and their hack prompt, used unmodified (Apache-2.0)
results/scores/         score files (.jsonl: one row per trajectory; .json: live rollouts and small summaries)
results/appendix/       label_split, faithfulness, probe_32b
runs/                   created at run time for logs and pod ids (gitignored)
```

Setup: `pip install -r requirements.txt`; `export OPENROUTER_API_KEY=...` (and `RUNPOD_API_KEY`, optionally `HF_TOKEN`, for the GPU launchers). To regenerate scores or run `attack/obfuscate.py --selfcheck`, clone Terminal Wrench to `./repo`; the `--analyze` modes do not need it. No key reaches a score file: the live runner unsets pod secrets before each agent command and redacts keys, SSH keys, emails, pod identifiers and IPs from observations. Run scripts from anywhere, e.g. `python scripts/attack/skeptic_test.py --analyze`.

| Paper | Script (`--analyze` recomputes from the shipped data, no key) | Data |
|---|---|---|
| Sec 3, Fig 2: the subset, pooled vs subset catch | `subset/patch_analysis.py`, `subset/scope_test.py` | `m0_full`, `m1_stripped`, `gemini_m1_full`, `results/episode_counts_*.json` |
| Sec 4: 96% to 4% on 209 hacks | `attack/skeptic_test.py --analyze`, `attack/iterate.py --analyze` | `skeptic_test`, `iterate` |
| Table 1: skeptic, paraphrase, other actor | `attack/skeptic_test.py --analyze`, `attack/obfuscate.py --analyze`, `attack/actor_generalize.py --analyze` | `obf_round1`, `actorgen_*`, `opus_m1_full_legit` |
| Table 2: defenses | `subset/table2_reproduce.py`, `defenses/*.py --analyze` | `defense_test`, `intentblind_test`, `consistency_test`, `adaptive_defense`, `reference_defense`, `*_baselines` |
| Sec 5, Table 5, App G, App I: live agents | `live/launch_live_agents.py --agent <model> --pool-only` (no key); `live/live_agents.py`, `live/env_sanity_sweep.py` (need `./repo` and keys) | `live_*_pooled.json`, `live_consolidated.json`, `env_sanity_sweep.json` |
| Sec 6, App H: probe, behavioral control | `probe/overt_malice.py --analyze`, `probe/mechanism_test.py --analyze`; `probe/capture_probe.py`, `probe/probe_controls.py`, `probe/probe32b_*.py` (need a pod and the captured activations) | `overt_malice`, `mechanism_test`, `results/internals_inputs_*`, `appendix/probe_32b/` |
| Sec 6.3: distillation | `sft/sft_data_prep.py`, `sft/launch_pod_sft.py`, `sft/pod_sft.py`, `sft/sft_score.py` | `results/sft_gen.json`, `sft_score.json` |
| App C, D: sensitivity, label split, faithfulness | `subset/scope_test.py`, `subset/label_split.py`, `faithfulness/coherence_test.py --analyze`, `faithfulness/adversarial_fab.py --analyze`, `faithfulness/faithfulness_judges.py --analyze`, `faithfulness/judge_crossfamily.py --analyze` | `appendix/label_split/`, `appendix/faithfulness/`, `coherence_test`, `adversarial_fab`, `faithjudge_*` |
| Prompts | obfuscator: `attack/obfuscate.py`; monitor rubrics: `common/monitor_rubrics.py`; live-agent instructions and reference judge: `live/live_common.py` | |

Live-run fields: `cond` (`B_innocent` = hack + narration instruction, `A_control` = hack only, `H_honest` = plain task), `verifier_passed`, `hollow_emit` (hack label, hollow at 0.5 or above), and the monitor's scores `emit_full`, `action_only`, `hidden_full`. Thresholds are the 95th percentile of each monitor's scores on that model's `H_honest` passes.
