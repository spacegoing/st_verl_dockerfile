# MDPLAN Review — Reading Order for Correctness Audit

Everything committed since you left, ordered by depth. Skim the top levels
for orientation; dive into the code files in Level 4 for correctness
audit. All file paths are relative to repo root
`/mnt/public/lichang93/st_verl_dockerfile`.

Git tags for stage-by-stage recovery:
- `mdplan-S0` — Phase-2 preview submission + master plan
- `mdplan-S1` — multi-domain infra
- `mdplan-S2` — plumbing (group_idx / domain_idx kwargs)
- `mdplan-S3` — multi-domain loss (V2/V3, hierarchical V4/V5, scatter_mean)
- `mdplan-S4-plan`, `mdplan-S5-partial` — deferred plans
- `mdplan-rename` — naming consistency + 2-node debug env
- mdplan-S6 in-flight (4 failed smoke iterations, 3 real bugs fixed)

## Level 1 — Orientation (5 min)

Read first to anchor the work.

1. **`bisimpo/dev_notes_mdplan_master.md`** — master plan: stage list,
   naming convention, cluster budget, legacy-combo crosswalk.
2. **`bisimpo/bspo_algorithm.tex`** — the formal spec the code
   implements. §1 single-domain (V1/V4/V5, already shipped pre-mdplan).
   §2 multi-domain (V1–V5, shipped by mdplan-S3).

## Level 2 — Math correctness (15 min)

Verify the equations the code must match.

3. **`bisimpo/bisim_eqs_multidomain.tex`**
   - §3 policy-ratio statistics `s_τ^±`, `s_g^±`, `s_dom^±` — check the
     mean-at-every-level form. "Why mean at every level" subsection is
     the load-bearing design note.
   - §4 clipping operators + the subsection *"Why Δ_z^(W) keeps the
     min(s_z⁺, s_z⁻) magnitude, and how sgn(s_z) is handled"* — walks
     through Wasserstein coupling rationale, impossibility of the
     `s − |X|` rewrite, sgn `.detach()` as principled, 7-row worked table.
4. **`bisimpo/bisim_eqs_recovery.tex`** — recovery-property proof:
   removing the `/|τ|` breaks recovery; A2 (single div at τ) and A3
   (mean at every level) both pass the unclipped-recovery test.
5. **`bisimpo/bisim_eqs_canonical_decomposition.tex`** — ANOVA-style
   argument that A3 is the unique canonical choice once you drop
   GRPO zero-mean assumption.
6. **`bisimpo/bisim_eqs_sign_analysis.tex`** — 3-variant sign-handling
   trade-off (detach / smooth surrogate / STE), on-policy
   pre-annihilation theorem, empirical measurement protocol.

## Level 3 — Stage dev notes (20 min)

Each mdplan stage has plan + execution log + exit-criteria check in one
doc.

7. **`bisimpo/dev_notes_mdplan_s1_infra.md`** — multi-domain k8s
   infrastructure: rayjob_md.yaml, submit_md.sh, entrypoint, Hydra md
   base config, combos. Exit criteria section summarises what S1
   shipped.
8. **`bisimpo/dev_notes_mdplan_s2_adv_approach.md`** — **key
   architectural decision doc**. "Arg-piercing vs in-place advantage
   estimator" trade-off analysis, decision rationale, and the minimal
   3-file diff that delivered plumbing.
9. **`bisimpo/dev_notes_mdplan_s3_md_loss.md`** — **main correctness
   doc for the multi-domain loss**. Math reference, code-change
   summary, synthetic-smoke output (all 5 variants). Read this
   carefully.
10. **`bisimpo/dev_notes_mdplan_s4_delta_est.md`** — (deferred) per-level
    δ derivation from dataset `d̂` — sketch only.
11. **`bisimpo/dev_notes_mdplan_s5_logging.md`** — (partially shipped)
    lists the 11 new hierarchical metrics that S3 already added to
    wandb; JSONL per-trajectory deferred.
12. **`bisimpo/dev_notes_mdplan_s6_smoke.md`** — 4-iteration debug log
    of the multi-domain smoke. Each failure has root-cause + fix
    documented. Useful as a "what can go wrong in the integration"
    checklist.

## Level 4 — Code (45 min — the actual correctness audit)

Read these files with the math in Level 2 next to you.

### 4a. BSPO loss (the heart of the correctness review)

13. **`verl/verl/trainer/ppo/core_algos.py`** — `compute_policy_loss_bspo`.
    Structure (line numbers approximate — grep for headers):
    - `_bspo_factorize_to_long_tensor` helper (np.unique inverse →
      int64 tensor). Accepts None, numpy, or torch tensor inputs.
    - Signature: adds `group_idx=None, domain_idx=None, **_kwargs`
      (backward compat).
    - Step 1–4 identical to single-domain (log_ratio, `s_τ^±`, `A_τ`
      recovery, `Δ_reg / Δ_w / pen` at τ).
    - **Multi-domain block (new in S3)**: gated on
      `variant in (hierarchy, strict_wasserstein) or lam_grp/lam_d > 0`.
      Inline `_scatter_mean(src, index, num_segments)` implementation.
      Computes `s_g^±` per group via scatter_mean, broadcasts to (B,),
      same for domain. Produces `Δ_g^(reg)`, `Δ_g^(W)`, `pen_g` and
      `Δ_dom^{...}` identically.
    - **Five branches**: simplest (unchanged), hierarchy (new), strict_w
      (new), w_penalty (extended with λ_grp·pen_g + λ_d·pen_dom when
      indices present), w_penalty_only (same extension).
    - Metrics dict: 12 original τ-level keys + 11 new hierarchical
      keys (`bspo_n_groups`, `bspo_n_domains`, δ_g_{reg,w}_mean,
      pen_g_{mean,active_frac}, δ_dom_*, `bspo_A_g_abs_mean`,
      `bspo_A_dom_abs_mean`).

14. **`verl/verl/workers/config/actor.py`** — `bspo_lambda_grp` and
    `bspo_lambda_d` fields added (both default 0, preserving
    single-domain behaviour).
15. **`verl/verl/trainer/config/actor/actor.yaml`** — same fields
    mirrored for Hydra.

### 4b. Plumbing (how `uid` / `data_source` reach the loss)

16. **`verl/verl/workers/actor/dp_actor.py`** — FSDP path. Around
    line 605–625: BSPO-gated `bspo_extra_kwargs` dict with
    `model_inputs.get('uid')` and `model_inputs.get('data_source')`
    forwarded as kwargs when `loss_mode == 'bspo'`. Non-BSPO losses
    unaffected.
17. **`verl/verl/workers/actor/megatron_actor.py`** — Megatron path.
    Two edit sites:
    - Around line 385–400: **prep step** — before
      `data.select(batch_keys=...)`, factorise
      `data.non_tensor_batch['uid']` and `['data_source']` via
      `np.unique(..., return_inverse=True)` into int64 (B,) tensors,
      attach to `data.batch` and append to `select_keys`. Gated on
      `loss_mode == 'bspo'`.
    - Around line 510–535: **loss_func call-site** — same BSPO-gated
      kwargs pattern as dp_actor.

### 4c. Multi-domain infra (skim if interested)

18. **`iter_kuberay_32nodes_verl_training/submit/rayjob_md.yaml`** —
    forked from sd template; diffs: `generateName: bra40-md-*`,
    `training-type: 40bra-md`, entrypoint path. Volcano + NCCL env
    identical.
19. **`iter_kuberay_32nodes_verl_training/submit/submit_md.sh`** —
    envsubst dispatcher. NNODES=2 → `k8s_b300_2node_debug` env.
    Regex allows hyphens in combo_id (`^c[a-z0-9-]+$`).
20. **`verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh`** —
    entrypoint. Gym on head + workers; Hydra `--config-name=40bra_16node_md`;
    `BSPO_DEBUG=1` env appends the debug override pack (1 step,
    val_max_samples=64, train_batch_size=8, lr_warmup=0).
21. **`verl/my_scripts/k8s/config/40bra_16node_md.yaml`** — md Hydra
    base. Diffs from sd: `train_files` → nemogym_blend;
    `pass_rate_key: nemo_pass_rate`; all 7 nemogym server_urls;
    `domain_balanced: true`.
22. **`verl/my_scripts/k8s/config/env/k8s_b300_2node_debug.yaml`** —
    2-node env. PP=2·EP=8·TP=1 = 16 GPU = 2 nodes. Halved batches.
23. **`verl/my_scripts/k8s/config/combo_40Bra.yaml`** — all combos.
    New md ones: `cbdg-md-v{1..5}-smoke` (S6), `cbmd-v{1..5}` (S7
    formal). `domains: ""` triggers curriculum_sampler's auto-discover
    mode.
24. **`bisimpo/submit_bspo_md.sh`** — per-combo hparam dispatch.
    `cbdg-*` combos auto-set NNODES=2 + BSPO_DEBUG=1.

## Level 5 — Sweep design docs (optional, if reviewing the plan)

25. **`bisimpo/bspo_sweep_plan.tex`** — 5-axis prioritised sweep
    (objective / sign / aggregation / trajectory-norm / loss_agg_mode).
    Reference for future phase-2/3 runs.
26. **`bisimpo/bspo_stage0_sweep.tex`** — Stage 0 matrix (8 sd + 5 md).
    Updated to the new naming scheme.

## Current smoke status (2026-04-19 ~14:03 UTC)

`cbdg-md-v1-smoke` progression:
- v1 FAIL: `KeyError: jd_pass_rate` → fixed `pass_rate_key: nemo_pass_rate`
- v2 FAIL: `No valid samples after filtering` → fixed `domains: ""`
- v3 FAIL: `AssertionError 16 % 64` (val path) → fixed `val_max_samples=64`
- v4 FAIL: `AssertionError 16 % 64` **(train path)** — *under investigation*

The v4 failure is in `megatron_actor.py:407` `make_minibatch_iterator`
call, with `mini_batch_size=self.config.ppo_mini_batch_size=32` per the
dumped config but the assertion reports `mini_batch_size=64`. Looks
like a Megatron DP/EP adjustment path that multiplies
`ppo_mini_batch_size` somewhere. Investigating now.

## What's been verified for correctness

- **Synthetic numerical correctness** (S3 dev notes, dated 2026-04-19
  08:32 UTC): `B=4, T=64` synthetic inputs run all 5 variants without
  error. Single-domain branches produce numerically identical outputs
  to pre-S3. Multi-domain branches produce sensible non-zero values.
- **Module imports + factorise helper** (S2 dev notes, 2026-04-19 08:15
  UTC): `compute_policy_loss_bspo` signature carries the new kwargs;
  `_bspo_factorize_to_long_tensor(['a','b','a','c','b'])` returns
  `tensor([0,1,0,2,1])`.
- **Single-domain baseline runs unaffected**: cbsp201/202/301/302 are
  still making progress at 6h+ on the mdplan-S2+S3 code (S2/S3 changes
  only activate when `loss_mode == 'bspo'` and the multi-domain kwargs
  are passed; sd runs go through the unchanged path).

## What's not yet verified

- End-to-end multi-domain smoke has not yet reached step 1 (v4 failed
  on train `make_minibatch_iterator` assertion before step 1 started).
- Therefore: `scatter_mean`-over-real-indices at multi-GPU scale,
  megatron prep-step's uid/data_source tensors actually flowing
  through `rearrange_micro_batches`, and gradient-flow on the
  hierarchical terms are all still pending.
