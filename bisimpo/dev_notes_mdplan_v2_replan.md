# BSPO Multi-Domain — Re-Plan v2 (from known-good base)

**Status:** plan-only, awaiting user review. Nothing deleted or committed
under this plan yet.

## Motivation for the reset

The v1 mdplan accumulated ~15 commits across 3 git repos. Several bugs
surfaced only during smoke tests (pass_rate_key, `domains="all"`,
val_max_samples divisibility, train_batch_size divisibility). The design
also drifted (sd and md mixed into one function — reverted in one of the
later commits but the intermediate state persists in history). Running
and queued rayjobs may have been compiled against partial / buggy
snapshots of the code.

We reset to the last known-good state and redo the work with a cleaner
git workflow, a tighter three-stage structure, per-stage debug scripts,
and 8-node debug nodes.

## Known-good base commits (do not modify during the reset)

| repo                                   | base commit | message                                   |
|----------------------------------------|-------------|-------------------------------------------|
| `st_verl_dockerfile` (main)            | `0a8d621`   | [New] sweep plan                          |
| `verl` (submodule)                     | `7dc6b1fa`  | [New] bspo single domain run              |
| `iter_kuberay_32nodes_verl_training`   | `5e7a492`   | [Finish] final kuberay volcano unified yaml |

All three repos reset their working branches to these commits at the
start of the reset. Current mdplan commits remain in reflog / tags for
recovery but no new work builds on them.

## Cleanup (before any code work)

1. **Delete all queued rayjobs** (cbsp-v5-l1e2, cbdg-md-*). These were
   submitted under the buggy mixed-function design and have outdated
   Hydra configs pinned via PFS file references.
2. **Delete all running single-domain Phase-2 preview rayjobs** (cbsp201,
   cbsp202, cbsp301, cbsp302). They started at a time when the verl
   source was at `7dc6b1fa`, but the editable PFS install means their
   running Python has whatever the files say *now* — a Ray worker
   restart after one of my later commits would have pulled in the buggy
   multi-domain code even on a single-domain run. Safer to kill and
   resubmit from the clean base.
3. **Phase-1 v2 (cbsp101/103/104)**: already deleted earlier after
   collapse. Skip.
4. **Multi-domain smoke (cbdg-md-v1-smoke)**: delete the current v5 /
   v6 queued job.

After cleanup: zero BSPO rayjobs, cluster fully idle. This is the
pre-condition for the reset.

## Branch strategy

Two branches, both forked from the known-good bases.

### Branch 1: `nemo_bspo_md_dev` (dev + debug, staged commits)

- Fresh branch from known-good base in each repo
- Three commits, one per stage (see Stages below)
- Each commit includes the **code changes + debug scripts used to
  validate that stage**
- Per-stage tags: `mdplan-v2-stage-1`, `mdplan-v2-stage-2`,
  `mdplan-v2-stage-3`
- This branch is the development record — it's how we get to a working
  multi-domain implementation with evidence that each stage passed a
  smoke.

### Branch 2: `nemo_bspo_md` (final, single-commit squash)

- Forked from the same known-good base
- Single squashed commit: merge `nemo_bspo_md_dev` using
  `git merge --squash` + `git commit`
- Contains **only the production code** — debug scripts and
  stage-specific MD notes stripped out during the squash
- Commit message: `[New] bspo multi-domain — V2/V3 + hierarchical V4/V5`
- This is what's published / reviewed / merged to origin.

## Three stages

Each stage has: scope, file list, debug script, expected output, exit
criteria.

### Stage 1 — Multi-domain infrastructure (k8s + Hydra + yaml)

**Scope:** everything that doesn't touch verl loss code. Get kuberay
submission path producing a valid RayJob that loads the nemogym_blend
training set, starts Gym on all nodes, completes a val-before-train
cycle using the stock single-domain `bspo` loss with the md dataset.
This stage confirms the infrastructure pipe works before any
multi-domain loss is introduced.

**Files** (all new):
- `iter_kuberay_32nodes_verl_training/submit/rayjob_md.yaml`
  — templated RayJob with `bra40-md-` prefix, multi-domain entrypoint.
- `iter_kuberay_32nodes_verl_training/submit/submit_md.sh`
  — envsubst dispatcher with NNODES={2, 4, 8, 16}.
  **Change from v1**: regex `^c[a-z0-9-]+$` to allow hyphens.
- `verl/my_scripts/k8s/config/env/k8s_b300_8node_debug.yaml`
  — 8-node debug env (see hparam justification below).
- `verl/my_scripts/k8s/config/40bra_16node_md.yaml`
  — md Hydra base. **Change from v1**: `pass_rate_key: nemo_pass_rate`
  from the start (was the v1 bug).
- `verl/my_scripts/k8s/config/combo_40Bra.yaml`
  — add combos `cbdg-md-v{1..5}-smoke` and `cbmd-v{1..5}`.
  **Change from v1**: `domains: ""` from the start (v1 had
  `domains: all` bug; empty string is auto-discover mode).
- `verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh`
  — entrypoint. BSPO_DEBUG=1 override pack designed for 8-node:
  `val_max_samples=256`, NO `train_batch_size` override (inherit env
  default `128` — avoids the v1 divisibility bug).
- `bisimpo/submit_bspo_md.sh`
  — per-combo dispatch, NNODES=8 default for `cbdg-*` combos.

**Debug script** (`bisimpo/test_stage1_infra.sh`):
- Runs envsubst + `kubectl create --dry-run=server` on the rayjob
  template to validate yaml syntax + combo_id regex.
- Submits `cbdg-md-v1-smoke` using the stock single-domain `bspo` loss
  (not `bspo_md`). This exercises: kuberay scheduling, volcano gang,
  Gym startup on 8 nodes, Hydra composition of
  `40bra_16node_md.yaml + combo_bank[cbdg-md-v1-smoke]`, curriculum
  sampler on nemogym_blend, dataset loading, model loading,
  val_before_train cycle.
- Asserts: rayjob SUCCEEDED OR val-core metrics appear.

**Exit criteria:** val-before-train produces per-source metrics in wandb.

### Stage 2 — verl infra plumbing (uid / data_source → loss fn)

**Scope:** thread the group/domain indices from
`non_tensor_batch` through the actor to a new loss function. This
stage does NOT implement the real `bspo_md` loss yet — just a stub
that asserts the kwargs arrive with correct shapes.

**Files**:
- `verl/verl/trainer/ppo/core_algos.py` — add:
  - `_bspo_factorize_to_long_tensor(arr, device)` helper.
  - **Stub** `@register_policy_loss("bspo_md")` that takes
    `group_idx`, `domain_idx` kwargs, runs `_bspo_factorize`, asserts
    non-None, logs `n_groups`, `n_domains` in metrics, and returns a
    zero loss (so gradients are zero and training is a no-op). The
    stub's real purpose is to **verify kwargs flow**, not to train.
- `verl/verl/workers/actor/dp_actor.py` — FSDP plumbing, gated on
  `loss_mode == 'bspo_md'`.
- `verl/verl/workers/actor/megatron_actor.py` — both prep step and
  loss_func call site, gated on `loss_mode == 'bspo_md'`.

**Debug script** (`bisimpo/test_stage2_plumbing.sh`):
- `docker exec vrl python3 -c "..."` unit check that
  `_bspo_factorize_to_long_tensor` converts `['a','b','a','c','b']`
  → `tensor([0,1,0,2,1])`.
- Submit `cbdg-md-v1-smoke` with
  `actor_rollout_ref.actor.policy_loss.loss_mode=bspo_md`.
- Assert step 1 completes, `actor/bspo_md_n_groups > 0`, and
  `actor/bspo_md_n_domains > 0` in wandb.

**Exit criteria:** stub smoke completes with `n_groups` and `n_domains`
populated; confirms plumbing works end-to-end.

### Stage 3 — `bspo_md` real loss (scatter_mean + 5 variants)

**Scope:** replace the stub with the real multi-domain loss.

**Files**:
- `verl/verl/trainer/ppo/core_algos.py` — replace stub with full
  `compute_policy_loss_bspo_md`. Inline `_scatter_mean`. 5 variants
  (simplest / hierarchy / strict_wasserstein / w_penalty /
  w_penalty_only). 11 `bspo_md_*` hierarchical metrics.
- `verl/verl/workers/config/actor.py` — add `bspo_lambda_grp`,
  `bspo_lambda_d` fields (default 0).
- `verl/verl/trainer/config/actor/actor.yaml` — mirror the same fields.

**Debug script** (`bisimpo/test_stage3_loss.sh`):
1. Synthetic unit test (`docker exec`): B=4, T=64 synthetic batch,
   run all 5 variants, assert pg_loss finite, `n_groups > 0` for
   hierarchy/strict_wasserstein.
2. Live smoke of `cbdg-md-v1-smoke` (V1 — baseline, simplest).
3. Live smoke of `cbdg-md-v2-smoke` (V2 — hierarchy, exercises
   scatter_mean over group).
4. Live smoke of `cbdg-md-v3-smoke` (V3 — strict_wasserstein,
   exercises scatter_mean over group + domain).

**Exit criteria:** all 3 live smokes SUCCEED, V2 and V3 show
`bspo_md_pen_g_active_frac` or `bspo_md_delta_g_w_mean ≠ 0` etc.

## Hyperparameter justification (debug vs formal)

### 8-node debug profile (`k8s_b300_8node_debug.yaml`, NEW)

Why 8 nodes instead of 2:
- **Init cost is roughly constant regardless of node count** (model load
  ≈ 80 s, Gym startup ≈ 30 s, Ray cluster init ≈ 60 s). Using 2 nodes
  saves no wall-clock on init; it only reduces compute throughput.
- **Single-node or 2-node DP=1 does not exercise DP gradient sync**.
  Multi-domain aggregation uses `scatter_mean` which in a DP setting
  runs independently on each DP rank; our debug must confirm it
  produces consistent gradients across DP>1. 2-node DP=1 would miss
  DP bugs.
- **Divisibility issues at small train_batch_size**. The v1 debug had
  `train_batch_size=8` which failed the `batch_size % mini_batch_size`
  assertion at scale. 8-node allows realistic `train_batch_size=128`
  (same as formal training) so debug validates the real path.
- **Only 3 debug nodes / 28 formal nodes** leaves room for 5 concurrent
  4-node formal runs (cbmd-v{1..5}): 5×4=20 + 8 debug = 28, 3 spare.

Computed shape:
- Total GPUs: 8 × 8 = 64
- Parallelism: PP=2, EP=8, TP=1 → replica = 16 GPUs, DP = 64/16 = **4**
- `ppo_mini_batch_size = 128` (matches standard 8-node settings)
- `ppo_micro_batch_size_per_gpu = 2`
- `log_prob_micro_batch_size_per_gpu = 4` → val batch quantum = 4×64 = 256

### `BSPO_DEBUG=1` override pack

Applied at entrypoint level when `BSPO_DEBUG=1`. Proposed values:

| override                                     | value | rationale                                      |
|----------------------------------------------|-------|------------------------------------------------|
| `trainer.total_training_steps`               | 1     | single training step → fast iteration          |
| `trainer.test_freq`                          | 1     | val at step 0 and step 1                       |
| `trainer.val_before_train`                   | true  | capture baseline val for diffing               |
| `trainer.save_freq`                          | 9999  | no checkpoint writes                           |
| `trainer.log_val_generations`                | 0     | no per-generation wandb spam                   |
| `data.val_max_samples`                       | 256   | divisible by 256 val batch quantum             |
| `data.train_batch_size`                      | *unchanged* (128 from env) | avoids divisibility assertions |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | 0 | no warmup for 1-step run                    |

### Formal `cbmd-v{1..5}` (4-node C11, matches sd Phase-1 v2 shape)

| setting                           | value | rationale                                     |
|-----------------------------------|-------|-----------------------------------------------|
| `NNODES`                          | 4     | matches Phase-1 v2 sd runs for comparability  |
| env                               | `k8s_b300_4node_c11` | unchanged from sd            |
| `total_training_steps`            | 120   | matches sd Phase-1 v2                         |
| `test_freq`                       | 10    | 13 val points across the run                  |
| `val_max_samples`                 | -1 (all 230 hard-eval rows) | three-source comparison per sd |
| `bspo_delta`                      | 3e-4  | same scale as sd Phase-1                      |
| `bspo_lambda_tj`                  | 1e-3  | same as sd Phase-1                            |
| `bspo_lambda_grp`                 | 0     | **default off**; the hierarchical penalty is optional and will be turned on in a later sweep  |
| `bspo_lambda_d`                   | 0     | same                                          |
| `ppo_epochs`                      | 2     | same as sd Phase-1                            |
| 5 runs × 4 nodes                  | 20 nodes | + 8 debug + 3 spare = 31 total exactly     |

Note on V2/V3: when `bspo_lambda_grp=0` and `bspo_lambda_d=0`, V2 still
exercises the scatter_mean block (it uses `Δ_g^(reg)` directly in
`ℓ_τ`, not just through the penalty). V3 requires `scatter_mean` over
both group and domain. So V2 and V3 are the "real" tests of the
hierarchical code path; V4 and V5 at `λ_grp=λ_d=0` are equivalent to
their single-domain versions from sd Phase-1.

## Timeline

- **Plan review (you)** — now
- **Cleanup**: delete all rayjobs → 5 min
- **Stage 1 dev**: 30 min (file forks)
- **Stage 1 debug smoke**: 40 min (8-node init + 1 step)
- **Stage 2 dev**: 20 min
- **Stage 2 debug smoke**: 40 min
- **Stage 3 dev**: 45 min (the real loss)
- **Stage 3 debug smokes**: 3 × 40 min (V1, V2, V3) = 120 min
- **Squash to `nemo_bspo_md`**: 15 min
- **Launch formal `cbmd-v{1..5}`**: 5 min (5 kubectl submits)

Total: roughly 6 h of active work + 2 h of smoke waiting = 8 h elapsed.
Formal runs then take ~20 h each (concurrent on 20 nodes).

## Open questions to confirm before starting

1. **Is the 8-node debug reserve correct?** The proposal assumes
   31 total = 20 formal (5 × 4) + 8 debug + 3 spare. Alternative: run
   formal on 8-node too (cbmd-v{1..5} × 8 = 40 > 31, won't fit
   concurrently). Please confirm 4-node formal is OK.
2. **Is keeping `λ_grp = λ_d = 0` for formal OK?** This means V4 and V5
   multi-domain become equivalent to their sd versions in practice; V2
   and V3 are the only variants that genuinely exercise hierarchy in
   formal. Alternative: set `λ_grp = 1e-3, λ_d = 1` (the paper defaults
   from `bspo_algorithm.tex` §2) so all 5 variants use the full
   hierarchy. I'd recommend the latter; the proposal above keeps zero
   only for conservatism.
3. **Do we also want a single-domain Phase-2 preview re-run?** Those 4
   runs (cbsp201/202/301/302) were near step 30 when they were
   implicitly affected by the editable-install issue. We can either
   resubmit them under the same names post-cleanup (using the known-good
   single-domain bspo code) or let them go. I'd recommend resubmitting
   since they're the controlled Phase-2 hparam sweep for the sd
   collapses.

## Files to be written during the reset (summary)

```
bisimpo/
├── dev_notes_mdplan_v2_replan.md           ← this file (plan)
├── test_stage1_infra.sh                    ← stage 1 debug script
├── test_stage2_plumbing.sh                 ← stage 2 debug script
├── test_stage3_loss.sh                     ← stage 3 debug script
└── submit_bspo_md.sh                       ← re-forked, 8-node default

iter_kuberay_32nodes_verl_training/submit/
├── rayjob_md.yaml                          ← new template
└── submit_md.sh                            ← new dispatcher

verl/my_scripts/k8s/
├── run_40bra_k8s_multi_domain.sh           ← new entrypoint
└── config/
    ├── 40bra_16node_md.yaml                ← new md Hydra base
    ├── combo_40Bra.yaml                    ← edit: +10 md combos
    └── env/
        └── k8s_b300_8node_debug.yaml       ← new 8-node debug env

verl/verl/
├── trainer/ppo/core_algos.py               ← +_bspo_factorize + bspo_md
├── trainer/config/actor/actor.yaml         ← +bspo_lambda_grp/d
├── workers/actor/dp_actor.py               ← BSPO_md-gated kwargs
├── workers/actor/megatron_actor.py         ← BSPO_md-gated prep + kwargs
└── workers/config/actor.py                 ← +bspo_lambda_grp/d
```

Count: 13 files total. About half are new, half are 3-10 line edits to
existing files.
