# Holistic analysis — what changed in verl to enable FiberPO

> Diff: `nemo` → `origin/fiberpo_ablation_single`, filtered to authors
> `spacegoing` and `lichang93`. Only the files relevant to the loss-function
> integration are listed; docs / shell scripts / plans-dir are skipped.

---

## 1. Files that matter for the integration

| File | Insertions | Role |
|---|---|---|
| `verl/trainer/ppo/core_algos.py` | +327 | the `compute_policy_loss_fiberpo` function + helper `_fiberpo_g_agg`, registered via `@register_policy_loss("fiberpo")` |
| `verl/trainer/config/actor/actor.yaml` | +37 | 10 new default fields: `fiberpo_epsilon_{pos,neg}`, `fiberpo_delta`, `fiberpo_C{,_pos,_neg}`, `fiberpo_clip_ratio_c`, `fiberpo_unclip_pos_floor`, plus 5 `fiberpo_ablate_*` flags |
| `verl/workers/config/actor.py` | +15 | mirror the yaml fields into the `ActorConfig` dataclass (typed defaults) |
| `verl/trainer/config/_generated_ppo_megatron_trainer.yaml`, `_generated_ppo_trainer.yaml` | — | auto-generated; inherit the new actor fields |

The rest of the spacegoing/lichang93 commits on that branch are documentation,
debug helper scripts, and curriculum-sampler work — not part of the FiberPO
loss integration.

## 2. Integration pattern

```
user config (YAML) ─► ActorConfig (dataclass) ─► loss function
 actor.yaml            verl/workers/config/actor.py     core_algos.py
```

Three touch points, each adds one line per new hyperparameter:

1. **YAML schema** (`actor.yaml`): default value + comment.
2. **Dataclass** (`ActorConfig` in `actor.py`): typed field with default.
3. **Loss fn** (`core_algos.py`): `config.get("fiberpo_delta", 0.4)` to read
   the value at runtime.

The registration with `@register_policy_loss("fiberpo")` at the top of the
loss function plugs it into verl's loss dispatcher. Selecting at runtime is
done by setting `actor_rollout_ref.actor.policy_loss.loss_mode=fiberpo` in
the combo yaml. No other plumbing is needed for a trajectory-level loss.

## 3. What the loss function takes / returns

```
def compute_policy_loss_fiberpo(
    old_log_prob: Tensor,      # (B, T)
    log_prob:      Tensor,     # (B, T)
    advantages:    Tensor,     # (B, T)  — GRPO-normalized, constant along t
    response_mask: Tensor,     # (B, T)
    loss_agg_mode: str = "token-mean",
    config:        Optional[ActorConfig] = None,
    rollout_is_weights: Optional[Tensor] = None,  # (B, T) or None
) -> tuple[Tensor, dict[str, Any]]:     # (scalar loss, metrics dict)
```

**Key fact for BSPO**: the signature does **NOT include `group_ids` or
`domain_ids`**. FiberPO works trajectory-level only (its `fiber_weight` is a
`(B,)` scalar multiplier), so it does not need them.

BSPO's Objectives 3 and 4 (trajectory-level only) would fit this signature
unchanged. **Objective 2 (Strict Wasserstein) requires group/domain IDs**
and will need either a signature extension or pre-computed aggregates
flowing in via `config`.

## 4. FiberPO's 9-step computation (for reference when writing BSPO)

```
Step 1: log_ratio = log_prob - old_log_prob, clamped to ±20
Step 2: log_s_pos, log_s_neg = per-trajectory means of clamp(±log_r, 0)
Step 3: fiber_w = exp( g_agg(log_s_pos, δ/2, T) − g_agg(log_s_neg, δ/2, T) )
Step 4: corrected_log_r = log_r − sign(log_r) · log_s_{same-sign}
Step 5: denom_log = −sign(log_r) · log_s_{opposite-sign}
Step 6: epsilon = ε_pos if A>0 else ε_neg  (per-token)
Step 7: clipped_ratio = exp(clip(corr, ±ε)) / exp(clip(denom, ±ε))
Step 8: per_token_obj = fiber_w · clipped_ratio · advantages
Step 8b/8c: dual-clip / unclip overlays for A<0,log_r>0 and A>0,log_r<0
Step 9: loss = -agg_loss(per_token_obj, response_mask, loss_agg_mode)
```

BSPO shares steps 1 and 2. From step 3 onwards it diverges:

- FiberPO: **multiplicative** `fiber_w` (exp of `g_agg` difference).
- BSPO: **additive** `Δ^(reg)` or `Δ^(W)` on top of the advantage term, no
  `g_agg` piecewise rollback, no per-token `corrected_log_r` / `denom_log`
  machinery.

BSPO is simpler in the per-token domain but more complex in the aggregation
domain (group + domain for Objective 2).

## 5. Hyperparameter additions to expect for BSPO

Mirror of the FiberPO triple: yaml default → dataclass field → loss-fn lookup.

Proposed names (see Q11 in `bspo_questions.md`):

```
bspo_variant                   # "simplest" | "strict_w" | "w_penalty" | "w_penalty_only"
bspo_delta_tau                 # trajectory clip threshold
bspo_delta_g                   # group clip threshold (Obj 2 only)
bspo_delta_dom                 # domain clip threshold (Obj 2 only)
bspo_lambda_tj                 # soft penalty coeff (Obj 3, 4)
bspo_use_arithmetic            # r-1 vs log r for s± (Q1)
```

Six fields vs FiberPO's ten. Simpler config surface.

## 6. Gotchas observed from FiberPO's history

- `fapo_tau_agg: null` with default fall-through to `2.0/delta` — a pattern
  where the YAML default is `null` and the loss fn computes the effective
  value from another hparam. Worth following for `C = delta/2` style
  relationships.
- `loss_agg_mode` is **ignored** inside FACPO (hardcoded to
  `seq-mean-token-mean`) because FAPO algorithms require that aggregation.
  BSPO Objectives 1/3/4 are per-trajectory scalars broadcast to per-token,
  so `seq-mean-token-mean` is also the right choice — see Q7.
- The `@deprecated` stub `compute_policy_loss` at `core_algos.py:1084` is
  PPO-classical; do NOT touch it. Register BSPO with a new name.

---

**Bottom line:** adding BSPO is structurally the same as FiberPO for
Objectives 1/3/4 (pure trajectory-level, fits the existing signature). For
Objective 2, we need to pipe `group_ids` + `domain_ids` from the data
pipeline through to the loss function — that is the one piece of
non-trivial plumbing and is the subject of Question 6 in
`bspo_questions.md`.
