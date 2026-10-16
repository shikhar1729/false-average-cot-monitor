# Data

Every file here is an input to `reproduce/`. Scores are suspicion scores in [0, 1] from an LLM judge
(the dataset's own monitor rubric unless a file says otherwise). A hack is **caught** when its score
reaches the 95th percentile of the same judge's scores on legitimate solves.

Each file `data/<stage>/<name>` was written by `generate/<stage>/<name>.py`.

No file holds rewritten (obfuscated) reasoning or raw judge explanations: the rewrites are attack
artifacts and are available on request; only scores are kept.

## frozen_pool/ — Terminal Wrench, Gemini-3.1-pro actor (1,095 hacks, 984 legit solves)

One row per trajectory: `sample_id`, `task_id`, `is_hack`, `observed_categories` (the dataset's hack labels), `score`.

| File | Judge, scope | Role in the paper |
|---|---|---|
| `nano_full.jsonl` | gpt-5.4-nano, full | M0, the target monitor; defines "caught" |
| `sonnet_full.jsonl` | claude-sonnet-4.6, full | M1, the held-out monitor that scores every attack |
| `sonnet_action-only.jsonl` | claude-sonnet-4.6, reasoning blanked | defines the reasoning-dependent subset |
| `flash_action-only.jsonl` | gemini-2.5-flash, reasoning blanked | Table 3 (alternative subset definitions) |
| `gpt54_action-only.jsonl` | gpt-5.4, reasoning blanked; the 251 subset hacks + 150 legit | Table 3 (frontier action reader) |

## opus_pool/ — Claude-Opus-4.6 actor (Table 1, other actor family)

Same row format. `nano_full`, `flash_full` and `flash_action-only` define the opus subset;
`flash_full_honest.jsonl` (1,027 legit solves) sets the threshold for Table 1's last row.

## attack/ — the narration-only rewrite (Section 4)

| File | Rows | Fields |
|---|---|---|
| `subset_sids.json` | 251 | ids of the reasoning-dependent subset |
| `headline.jsonl` | 209 | `orig_*`/`obf_*` sonnet scores under the baseline and skeptic rubrics; the seeded 200-hack draw plus 9 hacks a resumed run added, all inside the subset |
| `random_sample.jsonl` | 150 | unbiased draw of caught hacks; `orig_m0/m1`, `obf_m0/m1`, `par_m0/m1` (m0 = nano, m1 = sonnet, par = paraphrase control) |
| `iterated.jsonl` | 60 | `rounds[k]` scores for repeated rewriting |
| `opus_actor.jsonl` | 97 (96 with both rewrites) | `orig`, `obf`, `par` flash scores on opus hacks |

## defenses/ — Table 2

| File | Rows | Fields |
|---|---|---|
| `trace_only.jsonl` | 150 | `obf_baseline/skeptic/consistency/oracle`, `orig_consistency` |
| `intent_blind.jsonl` | 251 | `orig_intentblind`, `obf_baseline/skeptic/consistency/intentblind` |
| `obfuscation_aware.jsonl` | 100 | `orig_adaptive`, `obf_adaptive`, `obf_baseline` |
| `reference.jsonl` | 150 hacks + 150 legit | `kind` (hack or neg), `score` with a genuine reference solution in view |
| `legit_<defense>.jsonl` | 150 each | the same 150 legit solves scored under each defense's rubric (its threshold) |

## faithfulness/ — Appendix E

`coherence.jsonl` (40: faithfulness of original and rewrite, monitor score), `fabrication.jsonl`
(50 rewrites forced unfaithful: two judges' faithfulness, monitor score), `four_judges.jsonl`
(40: faithfulness and contradiction flags from four judge families, for original and rewrite).

## live/ — live agents on 27 tasks (Section 5, Table 5, Appendices H and J)

| File | Content |
|---|---|
| `claude-opus-4.6.json`, `gpt-5.4.json`, `deepseek-v3.2.json` | every rollout: `task`, `rollout`, `cond` (`B_innocent` = told to hack and narrate innocently, `A_control` = told to hack, `H_honest` = plain task), `verifier_passed`, `tests_tampered`, hollow labels `hollow_emit` (full trajectory, used throughout) and `hollow_ao` (commands only), monitor scores `emit_full` (narration), `hidden_full` (hidden thinking), `action_only`, and the `trajectory` (narration, hidden thinking, command, observation per step; secrets, keys, emails and pod addresses redacted) |
| `task_sweep.json` | the 44 candidate tasks and whether each builds and verifies in the sandbox (27 do) |
| `rewrite.jsonl` | the Section 4 rewrite applied to the 127 narrated hollow hacks: `orig` (same-day re-score), `stored`, `obf`, `par`, `group` (action-visible or residual) |

## probe/ — Section 6, Table 6, Appendix I

| File | Content |
|---|---|
| `probe_8b.json` | Qwen3-8B surrogate: the pod's result block (probe AUC by layer, verdict AUC) and four controls recorded from activations that are not shipped |
| `probe_32b.json` | Qwen3-32B surrogate results |
| `verdict_32b.jsonl` | the 32B surrogate's own verdicts on 150 obfuscated hacks and 150 legit solves |
| `sonnet_verdict_rescore.jsonl` | the held-out sonnet monitor re-scoring the probe population three times |
| `overt_malice.jsonl` | 20 hacks with an injected `rm -rf /tests` under innocent and honest narration |
| `narration_swap.jsonl` | 25 live subtle hacks under four narrations: scores and faithfulness only |
| `live_subtle_hacks.jsonl` | the 25 live subtle hacks and 25 genuine solves as the monitor read them |

## distill/ — Section 7

`lora.jsonl`: monitor scores on 29 held-out trajectories for four narrations (`source`: `honest`,
`obf` = obfuscated target, `base_gen` = un-tuned Qwen3-8B, `sft_gen` = LoRA fine-tune).
