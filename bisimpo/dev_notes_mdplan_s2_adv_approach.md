# MD-S2 — Arg-piercing vs in-place advantage-estimator decision

**Goal.** Decide the cleanest way to get `group_idx` (prompt id) and
`domain_idx` (data_source id) into `compute_policy_loss_bspo`, where they
are needed to compute the multi-domain aggregators
`s_g^\pm = scatter_mean_over_g(max(\pm s_\tau, 0))` and
`s_{dom}^\pm = scatter_mean_over_{dom}(max(\pm s_g, 0))`, and the
hierarchical advantage aggregates `\hat A_g`, `\hat A_{dom}`.

## What information flows where (status quo)

| quantity                 | where it lives                                    | who needs it                             |
|--------------------------|---------------------------------------------------|------------------------------------------|
| `old_log_prob` (B, T)    | mini-batch tensor                                 | `policy_loss_fn` ✓                       |
| `log_prob` (B, T)        | mini-batch tensor                                 | `policy_loss_fn` ✓                       |
| `advantages` (B, T)      | mini-batch tensor (broadcast of scalar `\hat A_τ`)| `policy_loss_fn` ✓                       |
| `response_mask` (B, T)   | mini-batch tensor                                 | `policy_loss_fn` ✓                       |
| `index` / `uid` (B,) int | `non_tensor_batch['uid']`                         | `compute_grpo_outcome_advantage` ✓; BSPO needs it but *does not currently receive it* |
| `data_source` (B,) str   | `non_tensor_batch['data_source']`                 | reward manager ✓; BSPO needs it          |
| `pass_rate`, `jd_pass_rate` (B,) | `non_tensor_batch['extra_info']` (json-decoded) | BSPO δ-estimator (S4)                    |

The GRPO advantage computation (`compute_grpo_outcome_advantage` in
`core_algos.py:267-330`) already takes `index` as a kwarg. It is
**not** propagated through `dp_actor.py` / `megatron_actor.py` into
`policy_loss_fn`, so `compute_policy_loss_bspo` has no access to it.

## Two candidate approaches

### A — Arg-piercing via optional `**kwargs`

Plumb `index` and `data_source` down through the actor's `policy_loss_fn`
call-site as **optional** keyword arguments. Add a `**kwargs` bag to the
signature of every `policy_loss_fn` implementation (backward compatible:
other losses simply ignore it). BSPO reads `kwargs.get('group_idx')`
and `kwargs.get('domain_idx')` and aggregates via `torch.scatter_mean`.

Edits required:
1. `verl/workers/actor/dp_actor.py` — extract `uid` and `data_source`
   from `mini_batch.non_tensor_batch` at the `policy_loss_fn` call-site;
   convert to torch int64 tensors on the correct device; pass as
   `group_idx=..., domain_idx=...` kwargs.
2. `verl/workers/actor/megatron_actor.py` — same as (1).
3. `verl/trainer/ppo/core_algos.py` — `compute_policy_loss_bspo`
   signature gains `group_idx=None, domain_idx=None` kwargs; the
   multi-domain branches (V2/V3/V4/V5-hierarchical) require them.

3 files edited. Zero impact on non-BSPO loss functions (they ignore the
new kwargs). Works cleanly with `DataProto`'s existing slicing — `uid`
and `data_source` are already tracked correctly per mini-batch.

### B — Write our own advantage estimator at the same module level

Register a new `@register_adv_est(...)` entry in `core_algos.py` next to
`compute_grpo_outcome_advantage` that precomputes *everything* BSPO
needs: `\hat A_\tau`, `\hat A_g`, `\hat A_{dom}`, `s_g^\pm`,
`s_{dom}^\pm`, group_idx, domain_idx. Pack these into either
(i) a multi-channel advantage tensor `advantages` of shape `(B, T, K)`,
or (ii) auxiliary per-trajectory scalars attached to `DataProto` via
`meta_info`.

Edits required if (i):
- New advantage estimator function.
- Refactor `advantages` consumers across verl to expect `(B, T, K)`. Every
  existing loss function breaks. Not viable.

Edits required if (ii):
- New advantage estimator function computes scalars and writes them into
  `batch.meta_info['bspo_aux']` (dict of tensors).
- `dp_actor.py` / `megatron_actor.py` must still extract from
  `meta_info` at the `policy_loss_fn` call-site and forward to the loss
  fn — which is **arg piercing with extra steps**.

Edits required if (iii — in-place replacement of `\hat A_\tau`):
- The BSPO-multi-domain advantage estimator re-uses the `advantages` slot
  to store the *final* per-trajectory objective contribution
  `\ell_\tau` directly (computed at advantage time rather than loss time).
- `compute_policy_loss_bspo` then just pulls `\ell_\tau` out, broadcasts,
  and calls `agg_loss`. No new kwargs, no arg piercing.
- **But:** this requires computing `s_\tau^\pm` (per-token ratio stats)
  at advantage time, *before the policy_loss_fn receives log_prob /
  old_log_prob*. Since the ratio requires the most recent policy output,
  this is not possible at advantage time — advantages are computed once
  per training step, while `log_prob` is re-computed each mini-batch.
  The $s_\tau$-dependent parts cannot be pre-baked. Not viable.

## Decision — take approach **A**, with minimal change

Approach A is **3 focused edits**, backward compatible, respects verl's
existing data flow. Approach B has either a viability problem (iii),
a breaking change (i), or is merely "arg piercing with extra indirection"
(ii).

**What we ship in S2:**

1. `dp_actor.py:608` — extract `uid` and `data_source` from the current
   micro-batch's `non_tensor_batch`; convert to `torch.int64` device
   tensors; pass to `policy_loss_fn` as keyword args.
2. `megatron_actor.py:511` — same as (1).
3. `core_algos.py` — add a small helper
   `_as_group_domain_idx(non_tensor_batch, device)` in a common scope;
   `compute_policy_loss_bspo` signature adds
   `group_idx: Optional[torch.Tensor] = None,
   domain_idx: Optional[torch.Tensor] = None` kwargs. For S2 they are
   *accepted and validated but not used* (the multi-domain branches
   that consume them are S3). This keeps S2 narrow: plumbing only, no
   loss-logic change.
4. All other registered loss fns (`vanilla`, `gpg`, `clip_cov`,
   `gspo_token`, `geo_mean`, ...) receive a **trailing `**kwargs`** that
   they ignore. Equivalent to a no-op for them.

## Implementation plan for S2

1. Grep-verify that `uid` and `data_source` are always present in
   `non_tensor_batch` for a BSPO-compatible batch (they should be —
   `uid` is used by GRPO advantage, `data_source` by reward manager).
2. Add the helper `_as_group_domain_idx` to `core_algos.py`.
3. Add the optional kwargs to `compute_policy_loss_bspo` signature;
   store them on a module-level `_CURRENT_BSPO_IDX` only as a sanity
   check for S3 (they will be used in S3 to compute s_g/s_dom).
4. Add trailing `**kwargs` to all other `@register_policy_loss(...)`
   functions that currently have a fixed signature, so the new kwargs
   can be passed uniformly from the actor without TypeError on
   non-BSPO losses.
5. Edit `dp_actor.py` and `megatron_actor.py` to extract and pass the
   indices.
6. Smoke: `docker exec vrl python3 -c "import verl.trainer.ppo.core_algos"`
   — confirm no import error; confirm the module-level
   registration picks up the updated signatures.
7. Run a minimal actor forward in a test cell (optional) to catch kwarg
   plumbing bugs before S3.
8. Commit with tag `mdplan-S2`.

## Why do plumbing before the loss logic (S3)

Separating S2 from S3 reduces the diff-review burden and makes it easy
to reason about failure modes. If S2 breaks the existing single-domain
BSPO runs (cbsp501 = V1 multi-domain which is just the same V1 loss,
only with extra unused kwargs), the issue is definitely in plumbing. If
S3 fails, the issue is in the new loss logic. Interleaving the two would
hide root causes.

## Exit criteria for S2

- [ ] `uid`, `data_source` extracted from `non_tensor_batch` in both
      actors, passed as `group_idx`, `domain_idx` kwargs.
- [ ] `compute_policy_loss_bspo` signature updated; existing single-domain
      variants (`simplest`, `w_penalty`, `w_penalty_only`) behave
      identically (they do not read the new kwargs).
- [ ] All other `@register_policy_loss(...)` functions accept `**kwargs`
      without TypeError.
- [ ] `docker exec vrl python3 -c "from verl.trainer.ppo import core_algos; core_algos.compute_policy_loss_bspo"`
      succeeds.
- [ ] Committed with tag `mdplan-S2` in verl submodule; master repo
      gets the dev-notes addition with tag `mdplan-S2`.

## Dev notes / execution log

**2026-04-19 08:15 UTC** — S2 executed.

Files edited:
1. `verl/trainer/ppo/core_algos.py`:
   - Added `_bspo_factorize_to_long_tensor(arr, device)` helper. Accepts
     `None`, numpy array (str/int), or torch Tensor. Returns `None` or
     factorised int64 tensor on device.
   - `compute_policy_loss_bspo` signature: added `group_idx=None`,
     `domain_idx=None`, `**_kwargs` trailing. Single-domain variants
     (`simplest`, `w_penalty`, `w_penalty_only`) ignore the new kwargs
     — behaviour identical to before.
2. `verl/workers/actor/dp_actor.py`:
   - At the `policy_loss_fn` call-site, when `loss_mode == "bspo"`,
     build `bspo_extra_kwargs = {"group_idx": model_inputs.get("uid"),
     "domain_idx": model_inputs.get("data_source")}` and spread into the
     call. Non-BSPO loss modes are unaffected.
3. `verl/workers/actor/megatron_actor.py`:
   - Same pattern at the loss_func-level call-site. For megatron, `data`
     at this layer is the micro-batch tensor dict, so `data.get("uid")`
     returns `None` until a prep step injects uid/data_source as batch
     tensors (scheduled for S3 when V2/V3/V4-hierarchical need them).
     For the single-domain BSPO variants still running on megatron
     (V1/V4/V5 simplest), this passing of None is a no-op.

## Simplification vs original S2 plan

Original S2 plan called for adding `**kwargs` to every
`@register_policy_loss(...)` function. Replaced with a narrower change:
only pass `group_idx` / `domain_idx` when `loss_mode == "bspo"`, so
other loss functions (vanilla, gspo, sapo, facpo, faspo, gpg, clip_cov,
kl_cov, geo_mean, cispo, bypass_mode) don't need any signature
modification. Lower diff; zero risk to existing loss paths.

## Import smoke

```
docker exec vrl python3 -c "
from verl.trainer.ppo.core_algos import compute_policy_loss_bspo, _bspo_factorize_to_long_tensor
import inspect
sig = inspect.signature(compute_policy_loss_bspo)
print('bspo sig params:', list(sig.parameters.keys()))
# factorise sample
import torch, numpy as np
x = np.array(['a', 'b', 'a', 'c', 'b'])
print(_bspo_factorize_to_long_tensor(x, torch.device('cpu')))
"
→ signature params include group_idx, domain_idx, _kwargs
→ factorise([a,b,a,c,b]) → tensor([0, 1, 0, 2, 1])
```

## Exit criteria

- [x] `uid`, `data_source` extracted and passed in dp_actor.py (FSDP)
- [x] Same pattern in megatron_actor.py (data-dict may return None for
      pure single-domain megatron; S3 will add the prep step when V2/V3
      variants land)
- [x] `compute_policy_loss_bspo` signature updated; single-domain
      variants pass through unchanged
- [x] No signature change to non-BSPO loss functions (kwargs only passed
      when loss_mode is bspo)
- [x] Import smoke passes
- [x] Committed with tag `mdplan-S2`

## Handover to S3

S3 implements the V2/V3 branches using `group_idx` / `domain_idx`. If S3
targets megatron (required for 40Bra), S3 must also add the prep step
in `megatron_actor.py:358-390` that converts
`data.non_tensor_batch['uid']` and
`data.non_tensor_batch['data_source']` into int64 tensors inside
`data.batch` before `data.select(...)`, so the loss_func receives them.
For FSDP paths (if any 40Bra run uses FSDP backend), the dp_actor.py
edit already plumbs everything needed.

