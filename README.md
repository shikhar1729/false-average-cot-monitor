# A False Average

Code and data for *A False Average: Pooled CoT-Monitor Accuracy Conceals a Reasoning-Dependent
Fragility* (Shikhar Shiromani and Leo Richter; NeurIPS 2026 Workshop on Agents in the Wild).

A chain-of-thought monitor's pooled accuracy hides a subset of reward hacks it can only catch from
the agent's reasoning. Rewriting only that reasoning, with every command and output byte-identical,
drops the monitor's catch rate on the subset from about 95% to 4%, while the pooled rate barely moves.

## Layout

```
reproduce/   recompute every number in the paper from data/ (free: numpy only, no API key)
data/        the scores behind every table and figure (see data/README.md)
generate/    the paid pipelines that produced data/ (OpenRouter, the Terminal Wrench clone, GPU pods)
monitoring/  the Terminal Wrench authors' monitor (Apache-2.0), unmodified
prompts/     the Terminal Wrench hack prompt (Apache-2.0), given to the live agents
```

## Reproduce the paper (free)

```bash
pip install numpy
python reproduce/all.py
```

It prints each number the paper reports beside the number the data gives, and exits non-zero if any
disagree. Each section also runs on its own:

| Script | Paper |
|---|---|
| `reproduce/s3_setup.py` | Section 3, Appendix B: the pool, the subset, thresholds, realized and split-half false-positive rates |
| `reproduce/s4_attack.py` | Section 4, Tables 1, 3 and 4: the rewrite, its robustness, the subset sensitivity, the label split |
| `reproduce/s5_defenses.py` | Section 5, Table 2, Appendix E: the defenses and the rewrite's faithfulness |
| `reproduce/s5_live.py` | Section 5, Table 5, Appendices H and J: live agents, the hidden channel, the rewrite on live hacks |
| `reproduce/s6_probe.py` | Section 6, Table 6, Appendix I: the probe and its behavioral controls |
| `reproduce/s7_distill.py` | Section 7, Appendix K: the LoRA distillation |

Two inputs cannot be recomputed from shipped files, and the scripts say so where they appear: the
hindsight audit of Section 4 needs the rewrite texts, and four Qwen3-8B probe controls need the
activations. Both are recorded values.

## Regenerate the data (paid)

```bash
pip install -r requirements.txt
git clone https://github.com/few-sh/terminal-wrench repo     # or set TW_REPO to an existing clone
export OPENROUTER_API_KEY=...                                 # and RUNPOD_API_KEY for GPU pods
python -m generate.attack.random_sample --selfcheck           # free: checks the rewrite rail
```

Run every generator from the repository root as a module. `data/<stage>/<name>` is written by
`generate/<stage>/<name>.py`; each script's docstring gives its command, inputs and rough cost.
Shared code lives in `generate/common/`: the dataset interface (`tw.py`), the judge rubrics
(`judge.py`), the attack (`obfuscator.py`), the samples every experiment draws (`subset.py`), and one
OpenRouter client (`openrouter.py`).

| Data | Command (`python -m ...`) | Needs | Rough cost |
|---|---|---|---|
| `frozen_pool/*`, `opus_pool/*` | `generate.frozen_pool.score_pool` (`--help` lists the runs) | key, clone | $2 to $50 per pool |
| `attack/random_sample` | `generate.attack.random_sample --batch` | key, clone | $30 |
| `attack/headline`, `defenses/legit_skeptic` | `generate.attack.headline` | key, clone | $60 |
| `attack/iterated` | `generate.attack.iterated` | key, clone | $8 |
| `attack/opus_actor` | `generate.attack.opus_actor` | key, clone | $25 |
| `attack/sanitized`, `causal`, `opus_gpt54` | `generate.attack.<name>` (pre-registered: `generate/attack/prereg_checks.md`) | key, clone | $7, $5, $15 |
| `defenses/trace_only`, `legit_consistency`, `legit_oracle` | `generate.defenses.trace_only` | key, clone | $40 |
| `defenses/intent_blind`, `legit_intent_blind` | `generate.defenses.intent_blind` | key, clone | $60 |
| `defenses/obfuscation_aware`, `legit_obfuscation_aware` | `generate.defenses.obfuscation_aware` | key, clone | $25 |
| `defenses/reference` | `generate.defenses.reference` | key, clone | $30 |
| `faithfulness/coherence`, `fabrication`, `four_judges` | `generate.faithfulness.<name>` | key, clone | $3 to $7 each |
| `live/task_sweep` | `generate.live.task_sweep` | clone, pod | pod time only |
| `live/<agent>` | `generate.live.launch --agent <model>` | key, clone, pods | $28 to $89 per agent |
| `live/rewrite` | `generate.live.rewrite` (pre-registered: `generate/live/prereg_rewrite.md`) | key, clone | $12 |
| `probe/live_subtle_hacks` | `generate.probe.live_subtle_hacks` | key, clone, pod | $21 |
| `probe/overt_malice`, `narration_swap` | `generate.probe.<name>` | key | $1 and $7 |
| `probe/probe_8b` | `generate.probe.prepare_inputs`, then `launch_pod_8b`, then `controls_8b` | key, clone, pod | $7 |
| `probe/probe_8b_action_only` | `generate.probe.action_only prepare`, `launch_pod_8b --inputs runs/probe/internals_inputs_ao.jsonl --no-verbal --tag ao`, `action_only collect` | clone, pod | under $1 |
| `probe/verdict_32b`, `probe_32b` | `generate.probe.verdict_32b`; `prepare_inputs_32b`, `launch_pod_32b --part A` and `B`, `analyze_32b` | key, clone, pods | $10 |
| `probe/sonnet_verdict_rescore` | `generate.probe.sonnet_verdict_rescore` | key, clone | $15 |
| `distill/lora` | `generate.distill.prepare`, `launch_pod`, then `lora` | key, pod | $5 |

"key" is `OPENROUTER_API_KEY`; "clone" is the Terminal Wrench clone; "pod" is a RunPod GPU via
`runpodctl` and `RUNPOD_API_KEY` (and `HF_TOKEN` for gated weights). Every pod launcher sets a
terminate-after backstop at creation and deletes the pod when it finishes or fails. Intermediates
that hold rewritten text, activations or pod logs go to `runs/` (not tracked).

## What is not released

- Rewritten (obfuscated) reasoning texts. They are attack artifacts; we share them on request.
- Raw judge explanations; only the scores are kept.
- Model activations and fine-tuning data for the probe and distillation experiments.

Live-agent observations had secrets, keys, emails and pod addresses redacted before release.

## License

MIT (see `LICENSE`). `monitoring/` and `prompts/` come from Terminal Wrench under Apache-2.0
(`monitoring/LICENSE-terminal-wrench`).
