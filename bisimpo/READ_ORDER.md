# BSPO Reading Order — for correctness review

Priority: check BSPO implementation correctness first, everything else after.
All paths relative to repo root `/mnt/public/lichang93/st_verl_dockerfile`.

## Current state snapshot (2026-04-19)

- **Phase-1 v2 running** (3 RayJobs on 4 nodes each, launched 2026-04-18 18:36 UTC):
  - `bra40-sd-cbsp101-4n-w42xk` — simplest (Obj 1), step ~40/120
  - `bra40-sd-cbsp103-4n-8jgzx` — w_penalty (Obj 3), step ~40/120
  - `bra40-sd-cbsp104-4n-w92mz` — w_penalty_only (Obj 4), step ~40/120
- Phase-1 **v1** (launched 17:30 UTC) was cancelled at 18:36 UTC after the eval-upgrade smoke passed. v1 used the old 125-row nemogym_math eval; v2 uses the new 230-row hard eval (BeyondAIME 100 + AIME 2025 30 + OlympiadBench 100).
- Monitors: `b9nukq81n` (terminal), `bgfvevn8d` (progress + errors).
- ETA for Phase-1 v2 completion: ~14:40 UTC 2026-04-19.

## Correctness review — read in this order

### 1. Math spec (what the loss should compute)

- **`bisimpo/bisim_eqs_annotated.tex`** — annotated derivation. Start with:
  - §5.4 (shipped variants — Obj 1 `simplest`, Obj 3 `w_penalty`, Obj 4 `w_penalty_only`)
  - §7 (Q13 derivation — why Obj 2 ≡ Obj 3 at λ=0 under GRPO, so Obj 2 not shipped)
  - §8 (shipped signature — exact function args / return shape)
  - §10 (hparam defaults table, δ=3e-4, λ=1e-3)
  - §12 (shipped-files traceability table — maps each math symbol to code line)
- **`bisimpo/bspo_questions.md`** — 14 design questions with resolutions. Q13/Q14 are the rigorous math derivations.

### 2. The loss code itself

- **`verl/verl/trainer/ppo/core_algos.py`** — search for `compute_policy_loss_bspo` (~167 lines added at the end). Check against §5.4/§8:
  - `@register_policy_loss("bspo")` decorator
  - Arithmetic `s+` / `s-` computation: `(max(r-1,0) * mask).sum(-1) / seq_lens`  (verifies 1/T token-mean)
  - Variant switch: `simplest` / `w_penalty` / `w_penalty_only`
  - `seq-mean-token-mean` aggregation (matches `Ê_τ` of the math)
  - `torch.sign(s).detach()` (non-differentiable sign)
  - Metrics dict: 12 `actor/bspo_*` diagnostic keys
- **`verl/verl/trainer/config/actor/actor.yaml`** — 3 new fields: `bspo_variant`, `bspo_delta`, `bspo_lambda_tj`
- **`verl/verl/workers/config/actor.py`** — same 3 fields as dataclass defaults (typed)

### 3. Reward manager patch (for the new hard-eval)

- **`verl/verl/workers/reward_manager/nemogym_server.py`** — search for `_MATH_SOURCES`. One-line change: the math branch was gated on `data_source == "nemogym_math"`; now a tuple that also includes `beyondaime`, `aime2025`, `olympiadbench`. All route to the same math judge server via `server_urls` config.
- **`verl/verl/experimental/reward_loop/reward_manager/nemogym_server.py`** — imports `_build_verify_body` from the workers copy, so the patch is automatically picked up here too (no second edit needed).

### 4. Config wiring (how the loss reaches the trainer)

- **`verl/my_scripts/k8s/config/40bra_16node_sd.yaml`** — Hydra base config for all BSPO runs. Check:
  - `defaults:` block: merges `ppo_megatron_trainer` + `env/*.yaml` + `combo_40Bra@combo_bank` + `_self_`
  - `data.val_files` → `eval_bspo_hard/eval.parquet` (230 rows)
  - `reward_model.reward_kwargs.server_urls` → 4 math-family URLs all → port 20006
- **`verl/my_scripts/k8s/config/combo_40Bra.yaml`** — combo definitions. Check the BSPO combos:
  - `cdbgbspo` (1 step, ppo_epochs=1, curriculum_total_steps=10) — debug only
  - `cbsp101/103/104` (120 steps, ppo_epochs=2, curriculum_total_steps=300) — formal Phase-1
  - Phase-2 slots reserved: `cbsp201-205`, `cbsp301-309`, `cbsp401-409`

### 5. Launch scripts

- **`bisimpo/submit_bspo.sh`** — combo_id → Hydra overrides mapper. Check:
  - Case block mapping each combo to `variant / delta / lambda / lr_warmup`
  - 4 BSPO overrides: `loss_mode=bspo`, `bspo_variant`, `bspo_delta`, `bspo_lambda_tj`
  - Ablation knobs: `test_freq=10`, `val_before_train=true`, `save_freq=9999`, `log_val_generations=0`
  - Delegates to production `iter_kuberay_32nodes_verl_training/submit/submit.sh`
- **`bisimpo/launch_phase1.sh`** — submits the 3 Phase-1 combos in sequence
- **`iter_kuberay_32nodes_verl_training/submit/submit.sh`** — production k8s submitter. No BSPO-specific logic; combo regex was relaxed to accept `c[a-z]*[0-9]*` names.

## Eval-upgrade stage (done during your sleep)

Read in order:

- **`bisimpo/dev_notes_eval_upgrade.md`** — decision rationale: why BeyondAIME + AIME 2025 + OlympiadBench; what was rejected and why.
- **`bisimpo/etl_eval_hard.py`** — the ETL (reads 3 HF datasets → writes `downloads/datasets/eval_bspo_hard/eval.parquet`)
- **`/mnt/public/lichang93/downloads/datasets/eval_bspo_hard/eval.parquet`** — output (230 rows; 3 distinct `data_source` values → 3 distinct `val-core/<src>/acc/mean@1` wandb keys)

Smoke result (before Phase-1 v2 launched): beyondaime=0.51, aime2025=0.667, olympiadbench=0.68 at step 0.

## Project progression / dev notes

Chronological:

- **`bisimpo/plan0.md`** — your original plan (input; not maintained by me).
- **`bisimpo/dev_notes_planning.md`** — initial design doc (callstack, hparam schema, ablation plan, timeline, go/no-go gates).
- **`bisimpo/dev_notes_implementation.md`** — code change log: 4 files edited + 2 new scripts.
- **`bisimpo/dev_notes_dev_debug.md`** — smoke test iteration log (v1 InterpolationKeyError fix, v2 lr_warmup fix, v3 SUCCESS) + orphan PodGroup cleanup.
- **`bisimpo/dev_notes_eval_upgrade.md`** — eval-upgrade stage (this new one, during your sleep).
- **`bisimpo/dev_notes_ablation.md`** — Phase-1 tracking (launch record + result table templates; will be filled when Phase-1 completes).

## Analysis tooling (ready for when Phase-1 completes)

- **`bisimpo/analyze_phase1.py`** — parses wandb offline .wandb files → per-combo CSVs + `summary.csv` + `summary.md`. CUTOFF_UTC=18:36 so only v2 runs are analyzed. Per-source acc (beyondaime / aime2025 / olympiadbench) reported.
- **`bisimpo/plot_phase1.py`** — CSVs → 3 PNGs:
  - `val_acc.png` (3-panel per-source val curves)
  - `train_signals.png` (6-panel: pg_loss, grad_norm, entropy, ppo_kl, s_tau_abs, per_traj_abs)
  - `bspo_diagnostics.png` (Δ_reg / Δ_w / clip frac / penalty_active / s+ vs s-)
- **`bisimpo/wandb_parse_offline.py`** — reference parser you provided (not BSPO-specific; my scripts mirror its pattern).

## Early signal from Phase-1 v2 (through step 40)

| step | combo | beyondaime | aime2025 | olympiadbench | pg_loss | grad_norm | s_tau_abs |
|---|---|---|---|---|---|---|---|
| 0 (smoke) | — | 0.51 | 0.667 | 0.68 | — | — | — |
| 10 | cbsp101 | — | — | — | (skipped) | — | — |
| 10 | cbsp103 | 0.61 | 0.70 | 0.64 | 0.0128 | 0.089 | 7.8e-4 |
| 10 | cbsp104 | 0.54 | 0.667 | 0.73 | 0.0128 | 0.112 | 5.8e-4 |
| 20 | cbsp101 | 0.53 | 0.70 | 0.73 | **0.0** | 0.051 | 6.7e-4 |
| 20 | cbsp103 | 0.47 | 0.70 | 0.70 | 0.0128 | 0.073 | 6.0e-4 |
| 40 | cbsp101 | 0.44 | 0.667 | 0.69 | **0.0** | 0.047 | **8.6e-3** |
| 40 | cbsp103 | 0.54 | 0.567 | 0.67 | 0.0119 | 0.086 | **5.2e-3** |
| 40 | cbsp104 | 0.55 | 0.733 | 0.73 | 0.0122 | 0.127 | 8.4e-4 |

Early observations (noisy, 40/120 steps):

1. **cbsp101 (simplest / Obj 1) at δ=3e-4 is broken.** `pg_loss` stays 0.0 → clipping saturates (s_tau_abs=8.6e-3 = 28× δ). Δ_reg loses magnitude info, carries only sign, provides no restoring force. All 3 val sources regressing on cbsp101 between step 20 and 40. **Phase-2 implication: must sweep δ ≥ 1e-2 for Obj 1 to carry real signal.**
2. **cbsp103 (w_penalty) stability degrading.** s_tau_abs jumped 9× (6e-4 → 5.2e-3) between step 20 and 40. Val oscillates. λ=1e-3 may be too weak to control ratio drift at δ=3e-4. Phase-2 should sweep λ up.
3. **cbsp104 (w_penalty_only) looking healthy.** s_tau_abs bounded (5.8e-4 → 8.4e-4 → ~10× less drift than 101/103). Steady val gains on all sources. Current front-runner; Phase-2 should grid around default hparams.

These are early and unstable. The winner is decided on final_step + best val, not step-40 snapshots.
