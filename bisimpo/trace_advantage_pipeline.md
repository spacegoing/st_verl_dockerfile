# verl Advantage / Objective / Ratio Pipeline — Code Trace

Code-only evidence. All paths relative to repo root. Line numbers correspond to the current checkout at the time this trace was taken.

## TL;DR

1. **Advantage is NOT divided by seq_len at computation time.** GRPO computes a scalar $\hat A_\tau = (R_\tau - \bar R_g)/\sigma_g$ per trajectory, then broadcasts it across tokens. BSPO's loss code explicitly *recovers* the scalar by dividing by seq_len (line 1795) — this is only a no-op round-trip since advantage is constant along $t$.
2. **The objective is divided by seq_len (as token-mean) and by global_batch_size (as seq-mean).** BSPO hard-codes `loss_agg_mode="seq-mean-token-mean"`. Because BSPO's `per_token_obj` is constant along $t$, the token-mean step is mathematically a no-op, and what remains is a `(1/B_global)` average across sequences.
3. **s_pos / s_neg at trajectory level are divided by seq_len.** That's the 1/|τ| in the math. Group-level and domain-level aggregation are **not implemented** in the current BSPO code — the shipped form is single-domain only. If/when multi-domain is added, the math spec (`bisim_eqs_multidomain.tex`) calls for SUM (not mean) at g and dom levels, so the asymmetry with τ is by design.

---

## Q1 — Advantage: divided by seq_len?

**Answer: No at the advantage-computation step. Yes in BSPO's per-trajectory recovery step (line 1795), and that recovery is a no-op round-trip because advantage is already constant along tokens.**

### Evidence — advantage computation

File: `verl/verl/trainer/ppo/core_algos.py`

```
267 def compute_grpo_outcome_advantage(
...
303     scores = token_level_rewards.sum(dim=-1)                     # (B,)   R_τ
...
323     for i in range(bsz):
324         if norm_adv_by_std_in_grpo:
325             scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
326         else:
327             scores[i] = scores[i] - id2mean[index[i]]             # Â_τ scalar
328     scores = scores.unsqueeze(-1) * response_mask                # (B, T) broadcast
329
330     return scores, scores                                         # advantages, returns
```

- **Line 303**: per-trajectory scalar reward via `sum(dim=-1)`. No `/ seq_len`.
- **Line 325**: GRPO normalization `(R - μ_g) / σ_g`. No `/ seq_len`.
- **Line 328**: scalar is broadcast to `(B, T)` via `unsqueeze(-1) * response_mask`. Every valid token of trajectory $i$ carries the same value $\hat A_{\tau_i}$; invalid tokens carry 0. Total = $\hat A_{\tau_i} \cdot |\tau_i|$.

The vectorized variant is identical semantics:

```
333 @register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
334 def compute_grpo_vectorized_outcome_advantage(...):
...
349     scores = token_level_rewards.sum(dim=-1)
350     g = as_torch_index(index, device=scores.device)
351     mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
352     if norm_adv_by_std_in_grpo:
353         scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
354     else:
355         scalars = scores - mean_g[g]
356     advantages = scalars.unsqueeze(-1) * response_mask
357     return advantages, advantages
```

### Evidence — BSPO-side recovery

File: `verl/verl/trainer/ppo/core_algos.py`

```
1786     seq_lens = response_mask.sum(dim=-1).clamp(min=1)                    # (B,)
...
1793     # ── Step 3: per-trajectory advantage (constant along t under GRPO) ──
1794     # TODO: how average is calculated
1795     A_tau = (advantages * response_mask).sum(dim=-1) / seq_lens          # (B,)
```

- Line 1795 divides the masked sum by `seq_lens`. Since `advantages[i, t] = Â_τ_i * mask[i, t]`, this gives `(Â_τ_i · |τ_i|) / |τ_i| = Â_τ_i`. It is a round-trip recovery, not an additional normalization, and matches the math literal $\hat A_\tau$.

**Consistency check**: a length-invariant $\hat A_\tau$ reaches the BSPO objective regardless of $|\tau|$.

---

## Q2 — Objective: when/where divided by seq_len / bsz while summing?

**Answer: The BSPO loss hard-codes `seq-mean-token-mean`, which divides by seq_len then by global_batch_size. Because `per_token_obj` is constant along $t$, the seq_len division cancels; what remains is a $1/B_{\text{global}}$ average.**

### Evidence — BSPO packs per-trajectory scalar into a per-token matrix

File: `verl/verl/trainer/ppo/core_algos.py`

```
1815     # ── Step 5: broadcast to per-token for agg_loss ──────────────
1816     per_token_obj = per_traj.unsqueeze(-1).expand_as(advantages)          # (B, T)
1817     pg_losses = -per_token_obj
1818
1819     # ── Step 6: aggregate (hard-code math-correct mode per Q7) ───
1820     pg_loss = agg_loss(
1821         loss_mat=pg_losses,
1822         loss_mask=response_mask,
1823         loss_agg_mode="seq-mean-token-mean",
1824         **config.global_batch_info,
1825     )
```

- Line 1816: `per_traj` (B,) is expanded to (B, T). Every valid token of trajectory $i$ carries `per_traj_i`; invalid tokens carry `per_traj_i` too but will be masked out by `response_mask` inside `agg_loss`.
- Line 1823: the aggregation mode is hard-coded (the `loss_agg_mode` kwarg on the function signature, line 1747, is explicitly ignored per its in-code comment `# ignored; hard-coded below`).

### Evidence — what `seq-mean-token-mean` does

File: `verl/verl/trainer/ppo/core_algos.py`

```
1025 def agg_loss(loss_mat, loss_mask, loss_agg_mode, dp_size=1,
...
1065     elif loss_agg_mode == "seq-mean-token-mean":
1066         seq_mask   = torch.sum(loss_mask, dim=-1)                              # per-seq token count |τ|
1067         seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)   # ← /|τ| token-mean
1068         seq_mask   = (seq_mask > 0).float()                                    # non-empty seq filter
1069         if global_batch_size is None:
1070             global_batch_size = seq_mask.sum()
1071         loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size   # ← /B_global seq-mean
```

- Line 1067: **divide by seq_len**. For BSPO, `loss_mat` is constant along $t$ (= `-per_traj_i`), so this reduces to `-per_traj_i * |τ_i| / |τ_i| = -per_traj_i`. Mathematically a no-op for BSPO, but it makes the division real for any other loss that varies across tokens.
- Line 1071: **divide by global_batch_size**. This is the "seq-mean" step: sum across sequences, divide by the total (global) batch size.

So the final BSPO loss is

$$
\hat J_{\text{BSPO}} \;=\; \frac{1}{B_{\text{global}}} \sum_{i=1}^{B} \text{per\_traj}_i \;\cdot\; \text{dp\_size}
$$

The `dp_size` multiplication is gradient-compensation for DDP averaging across ranks — mathematically the loss is still an unweighted mean per trajectory.

`global_batch_info` (the `**` kwargs at line 1824) is a dict injected by `verl/workers/utils/losses.py:103-106` containing `dp_size`, `batch_num_tokens`, `global_batch_size`, `loss_scale_factor`.

### Comparison — PPO vanilla uses the caller's `loss_agg_mode`

For reference (`verl/verl/trainer/ppo/core_algos.py:1241-1242`):

```
1241 pg_loss = agg_loss(
1242     loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
1243 )
```

PPO vanilla respects the configured `loss_agg_mode` (default `token-mean`, which is just total-token mean with no per-sequence normalization). **BSPO deliberately overrides** this to `seq-mean-token-mean` — this is what produces the $\hat{\mathbb{E}}_\tau$ outer expectation of the math, rather than PPO's $\hat{\mathbb{E}}_t$ over tokens.

### Caller (actor worker) — no additional scaling

File: `verl/verl/workers/actor/megatron_actor.py`

```
506     policy_loss_fn = get_policy_loss_fn(loss_mode)
...
511     pg_loss, pg_metrics = policy_loss_fn(
512         old_log_prob=old_log_prob,
513         log_prob=log_prob,
514         advantages=advantages,
515         response_mask=response_mask,
516         loss_agg_mode=loss_agg_mode,
517         config=self.config,
518         rollout_is_weights=rollout_is_weights,
519     )
...
536     stats["actor/pg_loss"] = pg_loss.detach().item()
537     policy_loss = pg_loss
```

No post-processing between `agg_loss` output and the `policy_loss` that goes into `.backward()`. What BSPO returns IS the loss.

---

## Q3 — s_pos / s_neg: divided by seq_len at τ / g / dom?

**Answer:**

- **Trajectory level (τ)**: **yes**, divided by `seq_len` (1/|τ|). This matches the math.
- **Group level (g)**: **not implemented**. Current BSPO code is single-domain only.
- **Domain level (dom)**: **not implemented**. Same reason.

Per the multi-domain spec (`bisim_eqs_multidomain.tex`, §3), the intended aggregation at g and dom is `SUM` (NOT averaged) — so when it ships, it will deliberately *not* divide by group or domain size. The asymmetry (τ averages, g and dom sum) is by design.

### Evidence — τ-level (shipped)

File: `verl/verl/trainer/ppo/core_algos.py`

```
1785     # ── Step 2: per-trajectory s+ / s- (arithmetic form, per Q1) ─────
1786     seq_lens = response_mask.sum(dim=-1).clamp(min=1)                    # (B,)    |τ|
1787     dev_pos  = torch.clamp(ratio - 1.0, min=0.0) * response_mask          # (B, T)  [r_t - 1]_+ · mask
1788     dev_neg  = torch.clamp(1.0 - ratio, min=0.0) * response_mask          # (B, T)  [1 - r_t]_+ · mask
1789     s_pos    = dev_pos.sum(dim=-1) / seq_lens                             # (B,)    s_τ^+ = (1/|τ|) Σ [r_t-1]_+
1790     s_neg    = dev_neg.sum(dim=-1) / seq_lens                             # (B,)    s_τ^- = (1/|τ|) Σ [1-r_t]_+
1791     s_tau    = s_pos - s_neg                                              # (B,)
```

- Line 1789/1790: `sum(dim=-1) / seq_lens` — matches exactly the math definition
  \[
    s_\tau^{\pm} = \tfrac{1}{|\tau|}\sum_{t\in\tau}\max(\pm(r_t-1),\,0).
  \]
- Line 1786 `.clamp(min=1)` guards against empty-sequence div-by-zero.

### Evidence — g and dom levels are NOT in the code

A grep over the BSPO function (`core_algos.py:1742-1858`) and the whole trainer directory confirms:

- No aggregation of `s_pos` / `s_neg` across trajectories within a group (no `scatter_mean`, no `segment_sum`, no group-index tensor).
- No domain-level aggregation (the only `data_source` touched by BSPO is at the reward-routing level in `nemogym_server.py`, not in `core_algos.py`).

Consistent with single-domain design: the shipped objectives (`simplest`, `w_penalty`, `w_penalty_only`) only use τ-level quantities:

```
1808     if variant == "simplest":
1809         per_traj = delta_reg * A_tau                                      # uses Δ_τ^(reg), Â_τ only
1810     elif variant == "w_penalty":
1811         per_traj = delta_w * A_tau - lam * penalty                        # uses Δ_τ^(W), penalty(τ)
1812     else:  # w_penalty_only
1813         per_traj = s_tau * A_tau - lam * penalty                          # uses s_τ, penalty(τ)
```

All three variants above are the τ-level collapse of the multi-domain forms in `bisim_eqs_multidomain.tex` §8. No g or dom terms appear.

---

## Consistency check — putting it together

| level | quantity | divided by? | code location | math ref |
|---|---|---|---|---|
| token → τ | $\hat A_\tau$ recovery | `/|τ|` | core_algos.py:1795 | round-trip over $\hat A_\tau \cdot |\tau|$ |
| token → τ | $s_\tau^{\pm}$ | `/|τ|` | core_algos.py:1789–1790 | $(1/|\tau|)\sum_t[\cdot]$ |
| τ → group | per_traj aggregation (loss side) | `/B_global` (seq-mean) | core_algos.py:1067+1071 via agg_loss | $\hat{\mathbb E}_\tau$ |
| τ → group | $s_g^{\pm}$ | **not implemented** | — | math says SUM (no div) |
| group → dom | $s_{dom}^{\pm}$ | **not implemented** | — | math says SUM (no div) |
| seq → token | broadcast `per_traj → per_token_obj` | expand_as (no div) | core_algos.py:1816 | no-op, then agg_loss re-averages |
| token → seq (agg) | loss_mat → seq_losses | `/|τ|` | core_algos.py:1067 | token-mean (no-op for BSPO) |

**Interpretation for the reviewer:**

- The τ-level `/|τ|` appears in three places: once to recover $\hat A_\tau$ (line 1795), twice for $s_\tau^{\pm}$ (lines 1789/1790). All three are consistent with the math's $1/|\tau|$ convention.
- At the objective-aggregation step the `/|τ|` happens again inside `agg_loss` (line 1067) but it is mathematically idempotent for BSPO — the per-token loss matrix is constant along $t$. It exists to keep the `agg_loss` signature uniform across all policy-loss variants (PPO vanilla / GSPO / etc. have non-constant per-token losses and genuinely need the division).
- The multi-domain extension will **not** add any further `/|g|` or `/|dom|` divisions. Per the spec, those aggregations are plain sums, and the dimensional balancing is done instead via the $T_g / T_{dom}$ scaling inside the advantage (`bisim_eqs_multidomain.tex` §7) and the per-level $\delta$ thresholds (`bisim_eqs_multidomain.tex` §5).
