# BSPO Dev Notes — Stage: Planning

> Written **before** any code. Per plan0.md: "always write an analysis
> / plan before u go next step". This file is the analysis + implementation
> plan for BSPO on verl.
>
> Started: 2026-04-18.

---

## 1. Scope (recap from plan0.md + bspo_questions.md)

- **Single-domain only.** Drop any term that is constant over a single-domain
  batch. Use constants for code vars.
- **Objectives in scope.** After the Q13 rigorous derivation, the ablation
  set is {Obj 1, Obj 3, Obj 4}. Obj 2 (Strict Wasserstein) reduces identically
  to Obj 3 with $\lambda=0$ under verl's GRPO full-group baseline, so it
  is covered by the $\lambda=0$ point of Obj 3's sweep.
- **$s^\pm$ form.** Arithmetic (`max(r − 1, 0)` / `max(1 − r, 0)`) per Q1.
- **Aggregation.** `seq-mean-token-mean` is the unique math-correct choice
  per Q7.
- **Clip threshold at GSPO scale**, not GRPO's 0.28/0.2. Per plan0.md.
- **Hparam transport**: yaml > bash (robustness with shell/Hydra
  special-char rules).
- **Variant switching**: explicit flag, kuberay-compatible.

## 2. Callstack design (plan0.md Task 3A)

### 2a. Minimum var set

For Obj 1, Obj 3, Obj 4 in single-domain, the loss function only needs the
existing verl loss-function arguments:

```
compute_policy_loss_bspo(
    old_log_prob:       Tensor,  # (B, T)
    log_prob:           Tensor,  # (B, T)
    advantages:         Tensor,  # (B, T)  — GRPO-normalized, broadcast per-token
    response_mask:      Tensor,  # (B, T)
    loss_agg_mode:      str,
    config:             ActorConfig,
    rollout_is_weights: Tensor | None,   # ignored per Q9
)
```

**No new vars needed.** Rationale:
- $s^+_\tau$, $s^-_\tau$, $s_\tau$, $\hat A_\tau$ all derive from
  (`log_ratio`, `response_mask`, `advantages`).
- $\hat A_g \equiv 0$ and $\hat A_\text{dom} \equiv 0$ under GRPO (Q13);
  the loss function never reads them.
- `rollout_is_weights` is ignored (Q9).

### 2b. Callstack impact

Zero plumbing changes. Only three files touched, same pattern as FiberPO:

| File | Change |
|---|---|
| `verl/trainer/ppo/core_algos.py` | new `compute_policy_loss_bspo` + registered via `@register_policy_loss("bspo")` |
| `verl/trainer/config/actor/actor.yaml` | add `bspo_*` default fields |
| `verl/workers/config/actor.py` | add `bspo_*` dataclass fields |

No edits to `dp_actor.py` or `megatron_actor.py`. The stash pattern
(trajectory_rewards + group_ids) is deliberately **not** implemented now;
we gain the Obj 2 semantics for free via the $\lambda=0$ point of Obj 3.

## 3. Loss implementation (plan0.md Task 3B)

### 3a. Hparam schema (yaml-first)

Config names chosen to minimize bash/yaml special-char hell:
- All lowercase with underscore separators.
- No digits in names that Hydra tends to mis-parse.
- Default values in yaml, typed defaults in Python `ActorConfig`.

```yaml
# actor.yaml additions
bspo_variant: simplest        # simplest | w_penalty | w_penalty_only
bspo_delta: 3.0e-4            # clip threshold at GSPO scale (per plan0.md)
bspo_lambda_tj: 1.0e-3        # soft penalty coeff (Obj 3, Obj 4)
```

Python dataclass additions (`ActorConfig`):
```python
bspo_variant: str = "simplest"
bspo_delta: float = 3.0e-4
bspo_lambda_tj: float = 1.0e-3
```

Three fields. That is all. Aggregation mode is hard-coded to
`seq-mean-token-mean` inside the loss (Q7). `sign(s)` is `.detach()` (Q8).
Arithmetic $s^\pm$ form is hard-coded (Q1). No other knobs.

### 3b. Variant selection

Flag `bspo_variant` is a string enum. Python `match` on the string. Invalid
values raise immediately — fail fast. No bash-level escape hell.

### 3c. GSPO-scale default for `bspo_delta`

Per plan0.md: "epsilon of clipping should be at gspo's scale, not grpo's
0.28 / 0.2 scale". GSPO paper uses $\epsilon \approx 3 \times 10^{-4}$ for
a log-ratio clip. Our $s^\pm$ uses arithmetic $(r − 1)$, which is linearly
equivalent to $\log r$ for $r \approx 1$. Start at $3 \times 10^{-4}$
and sweep.

Sweep candidates: $\{1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}, 3 \times 10^{-3}, 1 \times 10^{-2}\}$. Phase 1 (differentiating objectives) uses the default
$3 \times 10^{-4}$.

### 3d. Loss body (pseudo Python, to be coded)

```python
@register_policy_loss("bspo")
def compute_policy_loss_bspo(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode="seq-mean-token-mean",  # hard-coded below
    config=None, rollout_is_weights=None,   # rollout_is_weights ignored (Q9)
):
    # Hparams
    variant = config.get("bspo_variant", "simplest")
    delta   = config.get("bspo_delta", 3e-4)
    lam     = config.get("bspo_lambda_tj", 1e-3)

    # Step 1 — log ratio (B, T)
    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)

    # Step 2 — per-trajectory s+ / s- (arithmetic form, per Q1)
    ratio = torch.exp(log_ratio)
    T_tau = response_mask.sum(-1).clamp(min=1)
    s_pos = (torch.clamp(ratio - 1.0, min=0.0) * response_mask).sum(-1) / T_tau   # (B,)
    s_neg = (torch.clamp(1.0 - ratio, min=0.0) * response_mask).sum(-1) / T_tau   # (B,)
    s_tau = s_pos - s_neg                                                          # (B,)

    # Step 3 — per-trajectory advantage (constant along t under GRPO)
    A_tau = (advantages * response_mask).sum(-1) / T_tau                           # (B,)

    # Step 4 — variant-specific Δ
    if variant == "simplest":
        delta_reg = torch.clamp(s_pos, max=delta) - torch.clamp(s_neg, max=delta)
        per_traj = delta_reg * A_tau
    elif variant in ("w_penalty", "w_penalty_only"):
        s_min = torch.minimum(s_pos, s_neg)
        sign_s = torch.sign(s_tau).detach()                                        # Q8
        delta_w = sign_s * torch.clamp(s_min, max=delta)
        penalty = torch.clamp(s_min / max(delta, 1e-12) - 1.0, min=0.0)
        if variant == "w_penalty":
            per_traj = delta_w * A_tau - lam * penalty
        else:  # w_penalty_only
            per_traj = s_tau * A_tau - lam * penalty
    else:
        raise ValueError(f"unknown bspo_variant: {variant}")

    # Step 5 — broadcast to (B, T) for verl's agg_loss
    per_token_obj = per_traj.unsqueeze(-1).expand_as(advantages)
    if rollout_is_weights is not None:
        # Per Q9, we ignore it. But if upstream passes it, don't multiply
        # (this keeps the signature compatible).
        pass

    # Step 6 — aggregate (math-correct per Q7)
    pg_loss = -agg_loss(
        loss_mat=per_token_obj,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
    )

    # Step 7 — metrics
    metrics = {...}
    return pg_loss, metrics
```

### 3e. Metrics (mandatory for ablation diagnostics)

Per-step metrics to emit so wandb offline data can answer "which variant
wins":
- `actor/bspo_variant`: string, the variant used (for filtering).
- `actor/pg_clipfrac`: fraction of trajectories where `s_pos > delta` OR `s_neg > delta`.
- `actor/ppo_kl`: same as verl standard — `masked_mean(-log_ratio, mask)`.
- `actor/bspo_s_pos_mean`, `actor/bspo_s_neg_mean`, `actor/bspo_s_tau_mean`,
  `actor/bspo_s_tau_abs_mean` — detect saturation.
- `actor/bspo_delta_reg_mean` (Obj 1 only): per-traj $\Delta^{(\text{reg})}$.
- `actor/bspo_delta_w_mean` (Obj 3 only): per-traj $\Delta^{(W)}$.
- `actor/bspo_penalty_active_frac` (Obj 3, 4): fraction of trajectories
  where `min(s_pos, s_neg) > delta` (penalty biting).
- `actor/bspo_penalty_mean` (Obj 3, 4): mean penalty value.

## 4. Debug + ablation combos (plan0.md Task 4)

### 4a. Debug combo (1-step smoke)

```yaml
cdbg_bspo:
  description: "BSPO 1-step smoke test — verify loss runs without error"
  loss_mode: bspo
  bspo_variant: simplest
  bspo_delta: 3.0e-4
  bspo_lambda_tj: 1.0e-3
  ppo_epochs: 1
  total_training_steps: 1
  domains: nemogym_math
  curriculum_total_steps: 50
  save_freq: 9999
```

Submit via `0418/submit_debug.sh cdbg_bspo 4` (4 nodes, C11 profile, no
offload, gmu=0.4 — cheapest config). Expected ~40 min wall clock.

Success = no error + perf_log.jsonl written + metrics dict contains
`actor/bspo_*` keys.

### 4b. Ablation combo list (Phase 1: identify winning objective)

Three combos. All use identical settings except `bspo_variant`. 120 training
steps, test_freq=10, val_max_samples=128. 4 nodes (C11) per run.

```yaml
cbsp_101:
  description: "BSPO Obj 1 (simplest) — delta=3e-4"
  loss_mode: bspo
  bspo_variant: simplest
  bspo_delta: 3.0e-4
  bspo_lambda_tj: 1.0e-3       # unused for Obj 1
  ppo_epochs: 2
  total_training_steps: 120
  domains: nemogym_math
  curriculum_total_steps: 300
  save_freq: 9999

cbsp_103:
  description: "BSPO Obj 3 (w_penalty) — delta=3e-4, lambda=1e-3"
  loss_mode: bspo
  bspo_variant: w_penalty
  bspo_delta: 3.0e-4
  bspo_lambda_tj: 1.0e-3
  ppo_epochs: 2
  total_training_steps: 120
  domains: nemogym_math
  curriculum_total_steps: 300
  save_freq: 9999

cbsp_104:
  description: "BSPO Obj 4 (w_penalty_only) — delta=3e-4, lambda=1e-3"
  loss_mode: bspo
  bspo_variant: w_penalty_only
  bspo_delta: 3.0e-4
  bspo_lambda_tj: 1.0e-3
  ppo_epochs: 2
  total_training_steps: 120
  domains: nemogym_math
  curriculum_total_steps: 300
  save_freq: 9999
```

Three runs in parallel on 4n × 3 = 12 of 31 nodes (trivially fits, no
Inqueue wait). Expected wall clock per run at C11 config = ~18 h
(120 steps × ~540 s + eval overhead).

### 4c. Phase 2 (conditional) — sweep winner

After Phase 1 completes and a winner is identified:

- If Obj 1 wins: sweep `bspo_delta ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` → 4 new runs (default already covered).
- If Obj 3 wins: sweep `(bspo_delta, bspo_lambda_tj) ∈ {1e-4, 3e-4, 1e-3} × {0, 1e-3, 5e-3}` → 9 runs. `lambda=0` point also gives Obj 2 (Q13 identity).
- If Obj 4 wins: same sweep as Obj 3.

Phase 2 worst case: 9 concurrent runs on 36 of 31 nodes → Volcano queues,
2 waves × 18 h = ~36 h wall clock.

## 5. Timeline

| Stage | Wall time | Cluster load |
|---|---|---|
| Planning + coding + debug combo prep | ~30 min | 0 |
| Debug 1-step smoke | ~40 min | 4 nodes |
| Phase 1 ablation (3 runs, single wave) | ~18 h | 12 of 31 nodes |
| Phase 2 hparam sweep | ~18-36 h | 12-36 nodes (may queue) |
| Analysis + final report | ~1 h | 0 |
| **Total** | **~40-55 h** | — |

## 6. Risks + mitigation

| Risk | Mitigation |
|---|---|
| BSPO loss diverges / NaNs on first step | 1-step smoke catches this before the 18h ablation runs |
| `bspo_delta=3e-4` is too tight → all clips saturate → no learning signal | Metrics `bspo_s_{pos,neg}_mean` and `bspo_clipfrac` reveal this on the smoke test; we can bump to 1e-3 before the ablation if saturated |
| `sgn(s)` at s≈0 can be numerically noisy in Obj 3 | `torch.sign(s).detach()` — any non-zero result is a constant to the gradient; at exactly s=0 the term is zero anyway |
| Wandb offline parser does not find files | Use `wandb_parse_offline.py` sample; if it fails, fall back to parsing `perf_log.jsonl` directly (always on PFS under `<rayjob>/`) |
| Ablation runs OOM on C11 | C11 has 112 GiB headroom (Plan C §5); BSPO loss adds zero memory vs FiberPO. Low risk. |

## 7. Go / no-go gates

| Gate | Pass criterion | Action if fail |
|---|---|---|
| Code compiles + imports | `python -c "import verl"` succeeds | Fix syntax / naming errors |
| 1-step smoke completes | `perf_log.jsonl` written with `actor/bspo_*` metrics | Debug the loss function |
| Phase 1 runs complete | All 3 RayJobs → SUCCEEDED | Investigate per run, likely hparam issue |
| Winner is distinguishable | Best val metric ≥ 2 stderrs above next | Phase 2 hparam sweep; if still ambiguous, report as a tie |

## 8. File list

**To create:**
- `bisimpo/dev_notes_implementation.md` (stage: implementation)
- `bisimpo/dev_notes_dev_debug.md` (stage: debug)
- `bisimpo/dev_notes_ablation.md` (stage: ablation)
- `bisimpo/dev_notes_analysis.md` (stage: analysis, final)
- `bisimpo/implementation_plan.md` — this file's code snippets in a single
  executable reference

**To edit (verl source):**
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/config/actor/actor.yaml`
- `verl/workers/config/actor.py`
- `verl/my_scripts/k8s/config/combo_40Bra.yaml` — add cdbg_bspo + cbsp_*
  combos

**To not edit (important):**
- `verl/workers/actor/dp_actor.py` — stash pattern deliberately NOT used
- `verl/workers/actor/megatron_actor.py` — same

## 9. Proceeding

Next step: **implement the code changes**, then debug smoke. No more
questions will be raised; the answer set is complete.
