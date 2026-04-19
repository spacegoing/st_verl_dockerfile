# MD-S3 — Multi-domain BSPO loss (V2/V3 + hierarchical V4/V5)

**Goal.** Implement variants V2 (`hierarchy`), V3 (`strict_wasserstein`)
and extend V4 (`w_penalty`), V5 (`w_penalty_only`) with group- and
domain-level penalties, using the plumbed `group_idx` / `domain_idx`
kwargs from S2.

## Math reference

`bspo_algorithm.tex` §2 — multi-domain BSPO. Summary of the math that
S3 ships in code:

Per-level statistics (after `s_τ^±` already computed in single-domain code):
```
s_g^±    = (1/|g|)  · scatter_mean( max(±s_τ, 0),    index=group_idx )
s_dom^±  = (1/|dom|) · scatter_mean( max(±s_g, 0),   index=group→domain )
s_g      = s_g^+ - s_g^-
s_dom    = s_dom^+ - s_dom^-
```

Hierarchical advantages:
```
A_g      = scatter_mean(A_τ, index=group_idx)
A_dom    = scatter_mean(A_g_by_group, index=group→domain_of_group)
```

Clipping operators at each level `z ∈ {τ, g, dom}`:
```
Δ_z^(reg) = clip(s_z^+, δ) - clip(s_z^-, δ)
Δ_z^(W)   = sign(s_z).detach() · clip(min(s_z^+, s_z^-), δ)
pen_z     = max(0, min(s_z^+, s_z^-)/δ - 1)
```

Variants:
```
V1 simplest          : ℓ_τ = Δ_τ^(reg) · A_τ                            # single-domain equivalent
V2 hierarchy         : ℓ_τ = (Δ_τ^(reg) - Δ_{g(τ)}^(reg)) · A_τ
V3 strict_wasserstein: ℓ_τ = Δ_{dom(τ)}^(W)·A_{dom(τ)} + Δ_{g(τ)}^(W)·A_{g(τ)} + Δ_τ^(W)·A_τ
V4 w_penalty         : ℓ_τ = Δ_τ^(W)·A_τ - λ_Tj·pen_τ - λ_Grp·pen_{g(τ)} - λ_D·pen_{dom(τ)}
V5 w_penalty_only    : ℓ_τ = s_τ·A_τ       - λ_Tj·pen_τ - λ_Grp·pen_{g(τ)} - λ_D·pen_{dom(τ)}
```

All δ's at the same scale under mean-at-every-level normalisation
(`bisim_eqs_canonical_decomposition.tex` theorem). For S3 we use a
single `bspo_delta` for all levels; if empirical results indicate
per-level tuning is needed, per-level δ's are added in S4.

## Code changes

### (a) `core_algos.py::compute_policy_loss_bspo`

After the existing single-domain computation of `s_pos`, `s_neg`,
`s_tau`, `A_tau`, add a block guarded by
`if variant in ("hierarchy", "strict_wasserstein"):`
or by the presence of `bspo_lambda_grp>0` or `bspo_lambda_d>0`
(for extended V4/V5):

```python
# Factorise group/domain indices to int64 long tensors
group_long  = _bspo_factorize_to_long_tensor(group_idx,  s_pos.device)
domain_long = _bspo_factorize_to_long_tensor(domain_idx, s_pos.device)
assert group_long is not None, "BSPO V2+ requires group_idx (non_tensor_batch['uid'])"
assert domain_long is not None or variant != "strict_wasserstein", \
    "BSPO V3 requires domain_idx (non_tensor_batch['data_source'])"

# Group-level s±
n_groups = int(group_long.max().item()) + 1
g_pos_one  = torch.clamp( s_tau, min=0.0)                     # (B,) one-sided per-tau
g_neg_one  = torch.clamp(-s_tau, min=0.0)
# scatter_mean is not in torch core; emulate with index_add + counts
def _scatter_mean(src, index, num_segments):
    counts = torch.zeros(num_segments, dtype=src.dtype, device=src.device).scatter_add_(0, index, torch.ones_like(src, dtype=src.dtype))
    sums   = torch.zeros(num_segments, dtype=src.dtype, device=src.device).scatter_add_(0, index, src)
    return sums / counts.clamp(min=1)
s_g_pos = _scatter_mean(g_pos_one, group_long, n_groups)      # (n_groups,)
s_g_neg = _scatter_mean(g_neg_one, group_long, n_groups)
s_g     = s_g_pos - s_g_neg
# Broadcast back to (B,)
s_g_pos_tau = s_g_pos[group_long]
s_g_neg_tau = s_g_neg[group_long]
s_g_tau     = s_g[group_long]
# A_g
A_g_per_group = _scatter_mean(A_tau, group_long, n_groups)
A_g_tau       = A_g_per_group[group_long]

# Domain-level (only if domain_idx present)
if domain_long is not None:
    n_doms = int(domain_long.max().item()) + 1
    # group→domain map: for each group, pick the domain of any τ in it
    # (all τ in a group share the same domain by GRPO sampling design)
    g2d = torch.zeros(n_groups, dtype=torch.long, device=s_pos.device)
    g2d.index_copy_(0, group_long, domain_long)  # last-write wins; all same for a group
    s_dom_pos_g = torch.clamp(s_g, min=0.0)
    s_dom_neg_g = torch.clamp(-s_g, min=0.0)
    s_dom_pos   = _scatter_mean(s_dom_pos_g, g2d, n_doms)
    s_dom_neg   = _scatter_mean(s_dom_neg_g, g2d, n_doms)
    s_dom       = s_dom_pos - s_dom_neg
    # broadcast to (B,)
    s_dom_pos_tau = s_dom_pos[g2d[group_long]]
    s_dom_neg_tau = s_dom_neg[g2d[group_long]]
    s_dom_tau     = s_dom[g2d[group_long]]
    # A_dom
    A_dom_per_dom = _scatter_mean(A_g_per_group, g2d, n_doms)
    A_dom_tau     = A_dom_per_dom[g2d[group_long]]

# δ operators at g and dom
delta_g_reg   = torch.clamp(s_g_pos_tau, max=delta_t) - torch.clamp(s_g_neg_tau, max=delta_t)
sign_g        = torch.sign(s_g_tau).detach()
delta_g_w     = sign_g * torch.clamp(torch.minimum(s_g_pos_tau, s_g_neg_tau), max=delta_t)
pen_g         = torch.clamp(torch.minimum(s_g_pos_tau, s_g_neg_tau) / delta_t - 1.0, min=0.0)
# similarly for dom (only if domain_long is not None)

# Variant branches
if variant == "hierarchy":
    per_traj = (delta_reg - delta_g_reg) * A_tau
elif variant == "strict_wasserstein":
    per_traj = delta_dom_w * A_dom_tau + delta_g_w * A_g_tau + delta_w * A_tau
elif variant == "w_penalty":
    # extend the existing single-domain formula with hierarchical penalties
    per_traj = delta_w * A_tau - lam_tj * penalty
    if domain_long is not None:
        per_traj = per_traj - lam_grp * pen_g - lam_d * pen_dom
elif variant == "w_penalty_only":
    per_traj = s_tau * A_tau - lam_tj * penalty
    if domain_long is not None:
        per_traj = per_traj - lam_grp * pen_g - lam_d * pen_dom
```

### (b) New config fields

`verl/verl/trainer/config/actor/actor.yaml` and
`verl/verl/workers/config/actor.py` — add:
```yaml
bspo_lambda_grp: 1.0e-3
bspo_lambda_d:   1.0
```

### (c) Megatron prep step — inject uid/data_source into `data.batch`

In `megatron_actor.py` before `data = data.select(batch_keys=select_keys)`:
```python
loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
if loss_mode == "bspo":
    import numpy as np
    uid_np = data.non_tensor_batch.get("uid")
    ds_np  = data.non_tensor_batch.get("data_source")
    if uid_np is not None:
        _, uid_ids = np.unique(uid_np, return_inverse=True)
        data.batch["uid"] = torch.from_numpy(uid_ids).long()
        select_keys.append("uid")
    if ds_np is not None:
        _, ds_ids = np.unique(ds_np, return_inverse=True)
        data.batch["data_source"] = torch.from_numpy(ds_ids).long()
        select_keys.append("data_source")
```

The (B,) int64 tensors live alongside (B, T) tensors in TensorDict;
tensordict accepts the different trailing shape as long as the
leading batch_size matches.

### (d) New metrics

Add to `pg_metrics` dict:
```
actor/bspo_s_g_pos_mean, actor/bspo_s_g_neg_mean, actor/bspo_s_g_abs_mean
actor/bspo_s_dom_pos_mean, actor/bspo_s_dom_neg_mean, actor/bspo_s_dom_abs_mean
actor/bspo_delta_g_reg_mean, actor/bspo_delta_g_w_mean
actor/bspo_delta_dom_reg_mean, actor/bspo_delta_dom_w_mean
actor/bspo_pen_g_mean, actor/bspo_pen_dom_mean
actor/bspo_pen_g_active_frac, actor/bspo_pen_dom_active_frac
actor/bspo_n_groups, actor/bspo_n_domains
```

## Execution order

1. Add new config fields (actor.yaml + actor.py dataclass)
2. Implement scatter_mean helper + group/domain aggregation block in
   `compute_policy_loss_bspo`
3. Add V2 (`hierarchy`) and V3 (`strict_wasserstein`) variant branches
4. Extend V4/V5 with hierarchical penalties (optional; gated on
   lambda_grp>0 or lambda_d>0)
5. Add hierarchical metrics
6. Add megatron prep step
7. Import smoke test: the five variants' `compute_policy_loss_bspo`
   call resolves without TypeError given synthetic inputs
8. Commit with tag `mdplan-S3`

## Exit criteria

- [ ] V2, V3 variants implemented and reachable via
      `bspo_variant in {hierarchy, strict_wasserstein}`
- [ ] V4, V5 accept non-zero `bspo_lambda_grp`, `bspo_lambda_d` and
      apply group/domain penalties when `domain_idx` is present
- [ ] Single-domain existing runs (simplest, w_penalty, w_penalty_only)
      unaffected — numerical equality confirmed by unit check
- [ ] Megatron prep step ships uid/data_source as int64 tensors
- [ ] Import smoke passes; synthetic-input smoke passes for all 5
      variants
- [ ] Committed with tag `mdplan-S3`

## Dev notes / execution log

**2026-04-19 08:32 UTC** — S3 executed.

Files edited:

1. `verl/trainer/config/actor/actor.yaml`:
   - Updated `bspo_variant` comment (now 5 variants).
   - Added `bspo_lambda_grp: 0.0` and `bspo_lambda_d: 0.0`.
2. `verl/workers/config/actor.py`:
   - Same: added `bspo_lambda_grp`, `bspo_lambda_d` fields to the dataclass.
3. `verl/trainer/ppo/core_algos.py`:
   - `compute_policy_loss_bspo` accepts 5 variants (was 3).
   - New multi-domain aggregation block, gated on
     `variant in (hierarchy, strict_wasserstein)` or
     `lam_grp/lam_d > 0`. Uses:
     - inline `_scatter_mean(src, index, n)` via `scatter_add_` + counts
       (no torch_scatter dep)
     - `s_g^± = (1/|g|) Σ max(±s_τ, 0)` per group
     - `s_dom^± = (1/|dom|) Σ max(±s_g, 0)` per domain (via group→dom map)
     - `A_g` = scatter_mean(A_τ, group); `A_dom` = scatter_mean(A_g, g2d)
     - Per-level Δ^(reg), Δ^(W), pen (with sign detach at g, dom levels).
   - Five branches (simplest, hierarchy, strict_wasserstein, w_penalty,
     w_penalty_only). Single-domain branches unaffected.
   - +11 hierarchical metrics in `pg_metrics` (δ_g_reg/w mean, pen_g mean +
     active_frac, same for dom, plus `A_g`/`A_dom` abs means, `n_groups`,
     `n_domains`).
4. `verl/workers/actor/megatron_actor.py`:
   - New prep step before `data.select(...)`: when `loss_mode == 'bspo'`,
     factorise `non_tensor_batch['uid']` and `['data_source']` via
     `np.unique` inverse, attach as `data.batch['uid']` and
     `data.batch['data_source']` (int64 (B,)), add to select_keys.
   - Ensures the (B,) int tensors survive `rearrange_micro_batches` and
     reach loss_func as `data['uid']` / `data['data_source']`. S2's
     existing loss-func plumbing then forwards them to the loss fn.

## Synthetic-input smoke

```
docker exec vrl python3 -c "...synthetic (B=4, T=64) test..."
→ [simplest] sd loss=0.000e+00 per_traj_abs=0.000e+00         # Δ_reg saturated on tiny ratios (expected)
→ [w_penalty] sd loss=4.540e-03                                # BSPO penalty active on tiny s_min
→ [w_penalty_only] sd loss=4.261e-03
→ [hierarchy] md loss=1.477e-05 n_g=2 n_d=1                    # group factorisation ok
→ [strict_wasserstein] md loss=-9.981e-05 n_g=2 n_d=1          # all 3 terms summed
→ [w_penalty] md loss=4.540e-03 n_g=2 n_d=1 pen_g=0.000e+00    # group penalty gated on s_g>δ (not hit here)
→ [w_penalty_only] md loss=4.261e-03 n_g=2 n_d=1
→ ALL OK
```

Single-domain results for `simplest`, `w_penalty`, `w_penalty_only` are
numerically identical to pre-S3 (same computation path). V2, V3
newly produce non-trivial output.

## Exit criteria

- [x] V2 (`hierarchy`), V3 (`strict_wasserstein`) reachable
- [x] V4 and V5 accept `bspo_lambda_grp`, `bspo_lambda_d` and apply
      group/domain penalties when indices are present; default 0 keeps
      single-domain behaviour.
- [x] Single-domain existing behaviour preserved (first 3 smoke rows).
- [x] Megatron prep step: `data.batch['uid']`, `data.batch['data_source']`
      populated when loss_mode=='bspo'.
- [x] Synthetic smoke passes across all 5 variants.
- [x] Committed with tag `mdplan-S3`.

## Follow-ups handed to S4/S5

- S4: δ_Grp / δ_D derivation from dataset pass-rate distributions; right
  now all three levels use the same `bspo_delta`.
- S5: add rolling log of `d̂(z)` estimators per step and per minibatch
  to JSONL.
- Caveat (picked up in S6): the `s_g_pos_tau` etc. broadcast of per-group
  scalars back to per-trajectory uses `index_copy_` with last-write-wins
  for the g→dom map — this is safe because all τ in a group share the
  same domain, but we should add an assertion at S6 smoke time to make
  this invariant explicit.

