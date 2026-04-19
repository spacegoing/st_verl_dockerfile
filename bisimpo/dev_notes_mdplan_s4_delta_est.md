# MD-S4 — δ estimation from dataset (deferred)

**Status.** *Plan only — execution deferred until after S6 smoke + S7
formal runs land baseline data.* Rationale below.

## Goal (when we execute)

Per `bspo_algorithm.tex` §2 "Effective thresholds", δ at each hierarchy
level is derived from flat hyperparameters δ_D, δ_Grp, δ_Tj and a
dataset-dependent bisimulation-distance estimator `d̂(z)`:

```
d̂(z)              = E_{τ ∼ P(·|z)}[ |R(τ) − E_{τ'∼P(·|z)}[R]| ]
R_maxres(dom)     = max_{τ ∈ dom} |R(τ) − R̄_dom|
d̂_Grp(g)          = (T_g² / T_dom²) · d̂(g)

δ_dom             = d̂(dom) / R_maxres(dom) · δ_D
δ_g               = δ_dom · d̂_Grp(g) / Σ_{g' ∈ dom} d̂_Grp(g')² · δ_Grp
δ_τ               = δ_g · δ_Tj / (T_g · δ_Grp)
```

At this point S3 uses a single `bspo_delta` for all three levels,
consistent with the *mean-at-every-level* canonical decomposition
(all `s_z` are O(ε) on the same scale). S4 would replace this with the
above derivation, producing per-trajectory `δ_τ_i` that depend on
`dom(τ_i)`, `g(τ_i)`.

## Data source for `d̂`

Two candidate sources of `R(τ)`:

1. **Rollout reward** (live, per-step). `R(τ)` = reward returned by the
   Gym/nemo-server at each training step. `d̂(z)` computed over the
   current batch. Rolling moving-average across steps to stabilise.
2. **`pass_rate`** / **`jd_pass_rate`** from `extra_info` in the
   training parquet. Pre-computed offline signals. Available at
   advantage-compute time; no dependency on the current policy's
   rollouts. Stable but biased (captures the base model, not the current
   policy).

Decision (when we execute): use (1) with a moving-average window of 5
gradient steps — tracks current policy; smooth enough to be numerically
stable; no extra data-loading I/O. Log both (1) and (2) so the gap
between rollout-time and dataset-time distance is observable.

## Why deferred

Three reasons:

1. **S3 already ships multi-domain code that works with a single δ.**
   The current `bspo_delta` uniform-across-levels choice is mathematically
   defensible under the canonical decomposition argument
   (`bisim_eqs_canonical_decomposition.tex`, §3.1: all s_z are O(ε) on
   the same scale when mean-normalised).
2. **The estimator requires a moving state held by the trainer** (to
   track d̂(z) across steps), which adds a code-surface concern — better
   to add this AFTER we have baseline numbers from S7 runs without δ
   scaling, so the effect of the scaling is observable against a
   baseline.
3. **The `R_maxres / d̂(dom)` ratio enters in the denominator of
   `δ_dom`**, which at on-policy approaches 0/0. Needs careful
   implementation + numerical safeguards. Not S6-smoke-critical.

## Execution plan for when S4 is run

1. Add `bspo_delta_d`, `bspo_delta_grp`, `bspo_delta_tj` config fields
   (replacing the single `bspo_delta`).
2. Compute per-step `d̂(dom)`, `d̂(g)`, `R_maxres(dom)` from the current
   batch's rewards. Store in a `BSPOEstimator` state on the actor
   (moving average across steps).
3. Inside `compute_policy_loss_bspo`, look up per-τ `δ_τ_i` using the
   estimator state and the τ's group/domain.
4. Log `d̂_{dom,g}` distribution statistics (mean, std, min, max) to
   wandb.
5. Ship a smoke that compares single-δ vs per-level δ on the same combo.

## Sketch of the estimator state

```python
class BSPOEstimator:
    def __init__(self, momentum=0.9):
        self.d_hat_dom  = {}   # dom_id -> running mean of |R - R̄|
        self.d_hat_grp  = {}   # grp_id -> running mean of |R - R̄|
        self.R_maxres   = {}   # dom_id -> running max
        self.momentum   = momentum

    def update(self, rewards, group_ids, domain_ids):
        # group-level: per-group mean absolute deviation from group mean
        # domain-level: per-domain ... same
        # rolling EMA with self.momentum
        ...

    def get_delta(self, group_id, domain_id):
        # returns (δ_τ, δ_g, δ_dom) given flat δ_D, δ_Grp, δ_Tj hyperparameters
        ...
```

## Exit criteria (when this stage runs)

- [ ] Per-level δ derived from `d̂` + flat hyperparameters
- [ ] Estimator state persists across gradient steps (EMA)
- [ ] wandb log distribution stats of `d̂(dom)`, `d̂(g)`
- [ ] Backward-compat: setting `bspo_delta_tj=bspo_delta_grp=bspo_delta_d=bspo_delta_uniform`
      reproduces S3's single-δ behaviour (unit check)

Deferred: no tag or commit for S4 at this time. Will be scheduled after
S7 baseline runs complete.
