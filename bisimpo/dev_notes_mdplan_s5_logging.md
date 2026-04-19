# MD-S5 — Extended logging (partially deferred)

**Status.** *Partial execution.* S3 already ships all per-step wandb
metrics the stage originally called for (11 new hierarchical metrics
on top of the existing 12 trajectory-level ones = 23 total BSPO metrics
per step). The additional per-minibatch / per-trajectory JSONL
artefacts are deferred to post-S7.

## What S3 already delivered (live in wandb now)

Per gradient step (already logged by `compute_policy_loss_bspo` via
`pg_metrics`):

| key                                 | level        | meaning                                 |
|-------------------------------------|--------------|-----------------------------------------|
| `actor/bspo_s_pos_mean`             | trajectory   | mean of s_τ^+                           |
| `actor/bspo_s_neg_mean`             | trajectory   | mean of s_τ^-                           |
| `actor/bspo_s_tau_mean`             | trajectory   | mean of s_τ                             |
| `actor/bspo_s_tau_abs_mean`         | trajectory   | E[|s_τ|]                                |
| `actor/bspo_delta_reg_mean`         | trajectory   | mean of Δ_τ^(reg)                       |
| `actor/bspo_delta_w_mean`           | trajectory   | mean of Δ_τ^(W)                         |
| `actor/bspo_clip_pos_frac`          | trajectory   | frac of τ with s_τ^+ > δ                |
| `actor/bspo_clip_neg_frac`          | trajectory   | frac of τ with s_τ^- > δ                |
| `actor/bspo_penalty_active_frac`    | trajectory   | frac of τ with min(s+,s-) > δ           |
| `actor/bspo_penalty_mean`           | trajectory   | mean penalty value                      |
| `actor/bspo_per_traj_mean`          | trajectory   | mean ℓ_τ                                |
| `actor/bspo_per_traj_abs_mean`      | trajectory   | E[|ℓ_τ|]                                |
| `actor/bspo_n_groups`               | group        | # groups in current batch               |
| `actor/bspo_n_domains`              | domain       | # domains in current batch              |
| `actor/bspo_delta_g_reg_mean`       | group        | mean of Δ_g^(reg) broadcast to τ        |
| `actor/bspo_delta_g_w_mean`         | group        | mean of Δ_g^(W)                         |
| `actor/bspo_pen_g_mean`             | group        | mean group-level penalty                |
| `actor/bspo_pen_g_active_frac`      | group        | frac of τ with pen_g > 0                |
| `actor/bspo_delta_dom_reg_mean`     | domain       | mean of Δ_dom^(reg)                     |
| `actor/bspo_delta_dom_w_mean`       | domain       | mean of Δ_dom^(W)                       |
| `actor/bspo_pen_dom_mean`           | domain       | mean domain-level penalty               |
| `actor/bspo_pen_dom_active_frac`    | domain       | frac of τ with pen_dom > 0              |
| `actor/bspo_A_g_abs_mean`           | group        | E[|Â_g|]                                |
| `actor/bspo_A_dom_abs_mean`         | domain       | E[|Â_dom|]                              |

Reported every gradient step (default `actor/*` log cadence in verl's
trainer). At `test_freq=10` the val-core and val-aux per-data_source
metrics already land in the same wandb run.

## What this stage additionally called for (deferred)

1. **JSONL file at `<run_dir>/perf_log.jsonl`** — one line per
   minibatch with per-trajectory reward, ratio, s_τ, A_τ, pen_τ. Useful
   for off-policy retrospective analysis; not needed for live training
   decisions.
2. **`d̂_z` distribution per step** — intended to accompany S4's
   estimator. Deferred with S4.
3. **Per-minibatch granularity** (as opposed to the already-live
   per-gradient-step rollup).

## Why partially defer

- The per-gradient-step metrics from S3 are enough to observe the
  objective-variant differences and to diagnose saturation or drift
  in Phase-1 style analyses. S6 smoke and S7 formal runs can proceed
  without the JSONL layer.
- Per-minibatch JSONL adds I/O pressure on the PFS (one file handle
  per worker, appended many times per step). Needs careful buffering
  + flush strategy. Not trivial.
- The existing `<run_dir>/run.log` already captures stdout + per-step
  metrics dump from verl's trainer — sufficient for post-hoc analysis
  with `analyze_phase1.py`.

## Execution plan for when S5 is fully run

1. Add an `actor_rollout_ref.actor.bspo_jsonl_log: true` config flag.
2. When enabled, `compute_policy_loss_bspo` additionally writes a
   one-line JSON per call to `<run_dir>/bspo_step.jsonl` containing:
   ```json
   {
     "step": 42, "minibatch": 3,
     "per_tau": [
       {"tau_id": 0, "s_tau": 0.003, "A_tau": 1.1, "reward": 1.0,
        "group": 0, "domain": 0, "pen": 0.0},
       ...
     ]
   }
   ```
3. Post-analysis script `bisimpo/analyze_jsonl.py` reads the jsonl and
   produces scatter plots / histograms.

Deferred: no tag or commit for S5 at this time beyond this plan doc.
Will be scheduled alongside S4 or before a final production formal run
if per-trajectory diagnostics prove necessary.
