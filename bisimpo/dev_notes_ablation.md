# BSPO Dev Notes — Stage: Phase-1 Ablation

Phase-1 compares the 3 shipped BSPO objectives on a single configuration:
- Combos: `cbsp101` (Obj 1 simplest), `cbsp103` (Obj 3 w_penalty), `cbsp104` (Obj 4 w_penalty_only)
- Hparams: δ=3.0e-4, λ=1.0e-3
- Compute: 4-node C11 profile (no offload, gmu=0.4)
- Steps: 120, `ppo_epochs=2`, test_freq=10 → 12 val points + pre-train val at step 0
- Domain: `nemogym_math`

Goal of this stage: **rank objectives** on end-of-training math validation accuracy, with training-signal sanity checks (BSPO diagnostics within expected ranges).

---

## Launch record (v2 — with upgraded hard eval)

v1 launch at 17:30 UTC was **cancelled** at 18:36 UTC after the eval-upgrade smoke validated the new 230-row hard-eval parquet. v1 runs used the old 125-row `nemogym_math` eval; they were not allowed to complete. See `dev_notes_eval_upgrade.md`.

v2 launch:

| Combo | RayJob | Variant | Start UTC | Step-time (est.) | ETA UTC |
|---|---|---|---|---|---|
| cbsp101 | `bra40-sd-cbsp101-4n-w42xk` | simplest | 2026-04-18 18:36:51 | ~670s/step | ~14:40 2026-04-19 |
| cbsp103 | `bra40-sd-cbsp103-4n-8jgzx` | w_penalty | 2026-04-18 18:36:54 | ~670s/step | ~14:40 2026-04-19 |
| cbsp104 | `bra40-sd-cbsp104-4n-w92mz` | w_penalty_only | 2026-04-18 18:36:58 | ~670s/step | ~14:40 2026-04-19 |

ETA based on: 120 steps × 672s (ppo_epochs=2 → 2× update_actor over smoke) + 13 val points × 700s (230-sample eval × 2 passes per val) + 1 ckpt save at end. Overhead is larger than v1 because:
- v1 used 125 eval samples, v2 uses 230.
- v1 logged only `nemogym_math` acc; v2 logs 3 separate sources.

Monitors: `b9nukq81n` (terminal state), `bbh4pxuh3` (progress + per-source val + errors).

---

## Results table  <!-- populated by analyze_phase1.py -->

### Validation accuracy (math, 128-sample eval every 10 steps)

| combo | variant | final_step | last_val_step | last_val_acc | best_val_step | best_val_acc |
|---|---|---|---|---|---|---|
| cbsp101 | simplest | TBD | TBD | TBD | TBD | TBD |
| cbsp103 | w_penalty | TBD | TBD | TBD | TBD | TBD |
| cbsp104 | w_penalty_only | TBD | TBD | TBD | TBD | TBD |

### BSPO training diagnostics (mean over final 20% of steps)

| combo | pg_loss | grad_norm | entropy | s_tau_abs | Δ_reg | Δ_w | clip_pos | clip_neg | penalty_active | per_traj_abs |
|---|---|---|---|---|---|---|---|---|---|---|
| cbsp101 | TBD | TBD | TBD | TBD | TBD | — | TBD | TBD | — | TBD |
| cbsp103 | TBD | TBD | TBD | TBD | — | TBD | TBD | TBD | TBD | TBD |
| cbsp104 | TBD | TBD | TBD | TBD | — | — | TBD | TBD | TBD | TBD |

---

## Analysis checklist (apply once runs finish)

1. **Sanity**
   - All 3 hit step 120? If not, note the failure step and root cause.
   - `actor/pg_loss` non-NaN throughout?
   - `actor/grad_norm` finite and bounded (< 100)?

2. **Signal strength**
   - `bspo_clip_pos_frac + bspo_clip_neg_frac` < 30 %? If > 30 %, δ is too small (saturating).
   - `bspo_clip_*_frac` > 1 %? If < 1 %, δ may be too large (no regularization signal).
   - `bspo_per_traj_abs_mean` vs `actor/entropy`: per-traj should be ≤ advantage scale.

3. **Penalty activation (cbsp103/104 only)**
   - `bspo_penalty_active_frac` in [5 %, 50 %] → healthy λ regime.
   - If 0 %: penalty never fires (λ too small OR ratios never exceed δ). Usually means δ is too large.
   - If > 80 %: penalty always active, effectively a hard bound — try smaller λ.

4. **Winning criterion**
   - Primary: `best_val_acc` on math.
   - Tie-break: higher `last_val_acc` (plateau quality) if `best_val_acc` within 1.5 percentage points.
   - Secondary: lower `grad_norm` variance, less entropy collapse.

---

## Decision rule for Phase 2

- **If cbsp101 wins** (Obj 1 simplest):
  Phase 2 sweeps δ ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} → 5 runs, combos `cbsp201`–`cbsp205`.
  (No λ to sweep — Obj 1 has no penalty term.)

- **If cbsp103 wins** (Obj 3 w_penalty):
  Phase 2 sweeps δ × λ grid. Full 3×3 → 9 runs, combos `cbsp301`–`cbsp309`:
  δ ∈ {1e-4, 3e-4, 1e-3} × λ ∈ {1e-4, 1e-3, 1e-2}.

- **If cbsp104 wins** (Obj 4 w_penalty_only):
  Same 3×3 grid, combos `cbsp401`–`cbsp409`. (Obj 4 ≡ GRPO-like with bilateral Wasserstein penalty.)

- **If tie / no clear winner**:
  Top 2 variants advance to a reduced δ sweep (3 δ values each, 6 runs total).

---

## Next step after Phase 2

Populate `dev_notes_analysis.md` with:
- Final winning combo (variant + δ + λ)
- Evidence: best_val_acc, training-curve stability, significance vs FACPO baseline
- Recommendation for production use
