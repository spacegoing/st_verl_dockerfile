# Reading list — bspo_md correctness review

Ordered by priority. Start with Tier 1 items 1 + 2 side-by-side — that's
where correctness lives.

## Tier 1 — math ↔ code (the most important checks)

1. **Spec** — `bisimpo/bspo_algorithm.tex` §Multi-domain BSPO (lines 74–178)
   - Defines: `s_τ`/`s_g`/`s_d`, `d̂`, `δ_τ`/`δ_g`/`δ_dom`, `Δ^(reg)` / `Δ^(W)` / `pen`, and the 5 per-τ variants. This is the ground truth the implementation must match.

2. **Main loss** — `verl/verl/trainer/ppo/core_algos.py`
   - `_scatter_mean` helper — lines 1860–1879 (scatter_add + count-based mean; no torch_scatter dep).
   - `compute_policy_loss_bspo_md` — lines 1881–2053. Read top-to-bottom and cross-check against the spec section by section:
     - **Steps 1–2** (τ-level ratio + `A_τ`): match §Ratio statistics + §Advantages (GRPO branch).
     - **Steps 4–5** (`s_g`, `s_d` via MEAN-at-every-level): match §Ratio statistics for `s_g^±` / `s_d^±`.
     - **Step 6** (thresholds): **simplified** to flat `δ_τ = δ_g = δ_d = bspo_delta`. Paper's full effective-threshold recipe with `d̂(z)`, `d̂_Grp` is deferred — flag this.
     - **Step 7** (clipping operators): `_reg` / `_w` / `_pen` match spec §Clipping operators. `sign()` is detached per spec note (line 1953).
     - **Step 9** (variant branches): one-to-one with spec §Per-trajectory objective.
     - **Step 10** (aggregation): uses verl's `agg_loss(seq-mean-token-mean)` like the sd path.

3. **Advantage aggregation** — same file, lines 2005–2007 (`A_g`, `A_d`).
   - Currently uses plain `_scatter_mean(A_τ, group_idx, ...)`, not the token-weighted spec form `A_g = (1/T_g) Σ A_τ·|τ|`. Because `A_τ` is constant along tokens under GRPO (each τ has one advantage broadcast across its tokens), unweighted mean over τ equals token-weighted mean when all τ in a group have equal length. **If response lengths vary within a group, this deviates from spec** — worth discussing.

## Tier 2 — plumbing (are the kwargs what the loss expects?)

4. **Megatron path (primary for 40Bra)** — `verl/verl/workers/actor/megatron_actor.py`
   - `make_minibatch_iterator`, lines 376–395: keeps `uid` / `data_source` in `non_tensor_batch` when `loss_mode == 'bspo_md'`. Without this, they get dropped.
   - `forward_backward_batch` prep step, lines 440–457: reads `mini_batch.non_tensor_batch['uid']` + `['data_source']`, factorises to `(B, 1)` int64, attaches to `mini_batch.batch`.
   - `loss_func` kwargs handoff, lines 510–525: extracts `bspo_md_group_idx` / `bspo_md_domain_idx` from micro-batch, squeezes to `(B,)`, passes as kwargs.
   - **Correctness concern**: prep runs *after* `broadcast_dict_tensor` (line 425). This only works if every PP rank's `non_tensor_batch` contains `uid` / `data_source`. Multi-modal path works the same way, so the invariant is assumed by precedent, but I haven't independently verified the dispatch.

5. **FSDP path (mirror, unused by 40Bra runs but kept in sync)** — `verl/verl/workers/actor/dp_actor.py`
   - Lines 530–560: select keys, convert before `data.split`, inject `(B, 1)` tensors.
   - Lines 598–617: extract + pass kwargs.

6. **Helper** — `verl/verl/trainer/ppo/core_algos.py` lines 1860–1879
   - `_bspo_factorize_to_long_tensor` uses `np.unique(return_inverse=True)`. Order is deterministic per-rank but **not identical across DP ranks** — fine because scatter is rank-local.

## Tier 3 — where uid / data_source come from (upstream of the actor)

7. **uid injection** — `verl/verl/trainer/ppo/ray_trainer.py` line 1429: one `uuid` per *prompt*, then `gen_batch.repeat(n, interleave=True)` (line 1437) duplicates uid across the n responses. So all `n_resp_per_prompt` responses of one prompt share one uid → correct group semantics.

8. **data_source** — comes straight from the parquet column; no extra wiring needed. The `nemogym_blend/train_v2.parquet` has 6 domains, `val.parquet` has 5 (no `nemogym_math`).

## Tier 4 — config + launch surface (verify hparams wired correctly)

9. `verl/verl/workers/config/actor.py` lines 150–158 — `bspo_variant`, `bspo_delta`, `bspo_lambda_tj`, `bspo_lambda_grp`, `bspo_lambda_d`.
10. `verl/verl/trainer/config/actor/actor.yaml` lines 51–64 — defaults.
11. `verl/my_scripts/k8s/config/40bra_16node_md.yaml` — md Hydra base (`pass_rate_key=nemo_pass_rate`, `domain_balanced=true`, 7 Gym URLs).
12. `verl/my_scripts/k8s/config/combo_40Bra.yaml` lines 245–376 — cbdg-md-v{1..5}-smoke + cbmd-v{1..5} combos.
13. `bisimpo/submit_bspo_md.sh` — combo → {variant, δ, λ_Tj, λ_Grp, λ_D, NNODES} mapping. Formal λ values: **λ_Tj=1e-2, λ_Grp=1e-2, λ_D=1.0** (justification in the dev plan doc — only cbsp302 was stable).
14. `iter_kuberay_32nodes_verl_training/submit/{rayjob_md.yaml, submit_md.sh}` — RayJob template + dispatcher.
15. `verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh` — entrypoint + `BSPO_DEBUG=1` override pack.

## Tier 5 — evidence of what passed

16. `bisimpo/test_stage3_loss.sh` — synthetic unit test running all 5 variants (passes in-container), then live V1/V2/V3 smokes. Only on `nemo_bspo_md_dev` branch.
17. `bisimpo/dev_notes_mdplan_v2_replan.md` — plan + λ empirical justification + open questions (Q1/Q2/Q3). Dev branch only.
18. Live smoke wandb signals (cbmd-v1 step 1, 2026-04-19 18:42 UTC):
    - `bspo_md_n_groups = 46.4`
    - `n_domains = 5.08`
    - `s_τ_mean = 1.29e-3`
    - `pen_τ_mean = 19.7`
    - `pen_d_active_frac = 0.037`
    - `pg_loss = -1.58e-8`
    - `grad_norm = 0.19`
    - Step 1 ran cleanly.

## Known risks / things worth arguing about

- **Flat thresholds**: `δ_τ = δ_g = δ_d = bspo_delta`. Paper's `d̂(z)`, `R_maxres`, `T_g`-weighted recipe deferred. Under MEAN normalisation all levels share O(ε) scale, but this is an approximation worth a closer look.
- **A_g token-weighting**: see Tier 1 item 3. `_scatter_mean` ignores `|τ|` differences within a group.
- **Cross-DP group splitting**: shuffle is off everywhere (`actor.shuffle=false`, `data.shuffle=false`) and rollout repeat uses `interleave=True`, so the default DP split lands full groups on one rank. If a future config flips shuffle on, `s_g` / `s_d` become biased.
- **On-policy step 1 has near-zero deltas**: `log_ratio ≈ 0` (vLLM vs Megatron tiny numerical gap) saturates both sides of the clip → `Δ^(reg) ≈ 0`, `pg_loss ≈ 0`. Expected; divergence grows with training.

## Branch layout

- Main repo `b24f1f5`, verl `cb6c1f30`, iter_kuberay `db8856b` on `nemo_bspo_md` (single squashed commit each).
- `nemo_bspo_md_dev` preserves per-stage commits + all debug/test scripts + plan doc.
