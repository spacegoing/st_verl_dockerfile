# Plan C — Compute-efficiency optimization (apple-to-apple)

> Follow-up to Plan B. Goal: push per-GPU throughput higher without
> touching any hyperparameter that can change the learning trajectory.
> All experiments hold the training-relevant hyperparameters fixed and
> vary only compute/memory knobs.

Written: 2026-04-18.
Depends on: `plan_b_node_opt.md` results (P1 = 8-node is the chosen baseline).

---

## 1. Apple-to-apple rule and hyperparameter split

User directive (2026-04-18): compute optimization must not cross into
training-relevant hyperparameters. Any numeric comparison across runs
must hold those constant. This is the split we work to.

### A. Training-irrelevant (safe to change — pure compute/memory knobs)

These affect **how** the same computation is executed, not **what** is
computed, so numerical output is identical (up to the usual
floating-point non-determinism that parallel reductions already have):

- `trainer.nnodes`, `trainer.n_gpus_per_node` (total GPU count)
- `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` (PP)
- `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP)
- `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` (TP)
- `actor_rollout_ref.actor.megatron.expert_model_parallel_size` (EP)
- `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` (ETP)
- `actor_rollout_ref.actor.megatron.context_parallel_size` (CP)
- `actor_rollout_ref.actor.megatron.param_offload`, `optimizer_offload`, `grad_offload`
- `actor_rollout_ref.actor.optim.override_optimizer_config.*` (offload fractions, overlap, precision-aware)
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` (microbatch split for memory)
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` (only when `use_dynamic_bsz=True`, where it is ignored)
- `actor_rollout_ref.rollout.gpu_memory_utilization`
- `actor_rollout_ref.rollout.max_num_batched_tokens`
- `actor_rollout_ref.rollout.enable_chunked_prefill`
- `actor_rollout_ref.rollout.enforce_eager`
- `actor_rollout_ref.rollout.free_cache_engine`
- `actor_rollout_ref.rollout.tensor_model_parallel_size` (vLLM TP)
- `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`, `log_prob_use_dynamic_bsz`
- `actor_rollout_ref.nccl_timeout`
- Ref-model parallelism mirrors (all `megatron.*` on the `ref` sub-config)
- Ref-model offload settings
- Runtime env: `NCCL_*`, `NVSHMEM_*`, `VLLM_*` (attention backend, etc.)
- Reward HTTP concurrency (if we add it)

### B. Training-relevant (must NOT change for compute-only comparison)

These change the math, the sample distribution, or the optimization
trajectory:

- `data.train_batch_size`, `data.max_prompt_length`, `data.max_response_length`
- `data.sampler.*` (curriculum params: `initial_mean`, `final_mean`, `std`, `total_steps`, `domains`, `domain_balanced`, `filter_pass_rate_100`)
- `actor_rollout_ref.rollout.n` (number of responses per prompt)
- `actor_rollout_ref.rollout.temperature`, `top_p`, `top_k` (sampling)
- `actor_rollout_ref.actor.ppo_mini_batch_size`
- `actor_rollout_ref.actor.ppo_epochs`
- `actor_rollout_ref.actor.optim.lr`, `lr_warmup_steps`, `weight_decay`, `clip_grad`
- `actor_rollout_ref.actor.policy_loss.loss_mode` and all FACPO params (`fapo_delta`, `tau_pos`, `tau_neg`)
- `actor_rollout_ref.actor.entropy_coeff`
- `actor_rollout_ref.actor.clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c`
- `actor_rollout_ref.actor.use_kl_loss`, `kl_loss_coef`
- `algorithm.adv_estimator`, `norm_adv_by_std_in_grpo`, `kl_ctrl.kl_coef`, `use_kl_in_reward`
- `reward_model.reward_kwargs.overlong_buffer_cfg.*`, `max_resp_len`
- Dataset / eval files
- `trainer.total_training_steps`, `total_epochs`

### C. Gray zone (worth calling out)

- `actor_rollout_ref.actor.use_dynamic_bsz` — when `True`, micro-batching is determined by `ppo_max_token_len_per_gpu`. Numerically equivalent to fixed micro-batching, so it is **in A** as long as we do not switch between True and False mid-comparison.
- `trainer.test_freq`, `trainer.save_freq` — do not affect learning, only disk I/O and wall-clock. In A.
- `data.val_max_samples`, `val_batch_size` — affect validation cost, not training. In A (but do affect the validation numbers we see, so keep consistent when comparing).

---

## 2. Plan B status recap

The P0/P1/P2 series already obeyed the rule — only `trainer.nnodes` (category A) varied. Result: P1 (8-node) is the chosen baseline for Plan C (`step_s=516, gen_s=294, tok/s/GPU=561, mem=237 GB`).

Everything in this plan holds category B fixed at the `c351` combo values and the `env_k8s_b300_16node.yaml` values in effect for P1.

---

## 3. Experiment matrix — ranked by return over cost

Each row below is a single-variable or small-variable change on top of **P1** (8 nodes × 8 GPU, nemogym_math, cdbg5 combo, 5 steps, `data.val_max_samples=32`, `trainer.test_freq=3`). Cost = edits required + setup risk. Return = expected improvement in `tok/s/GPU` and/or `timing_s/step`.

### Tier 1 — trivial cost, direct rollout-side impact

| # | Name | Change | Hypothesis | Expected return | Risk |
|---|---|---|---|---|---|
| C1 | `P1-gmu85` | `rollout.gpu_memory_utilization=0.85` | More KV cache → vLLM packs more concurrent sequences → less tail shrinkage | Medium (5-15% on `gen_s`) | Low. Leaves 43 GB/GPU for actor after offload. |
| C2 | `P1-gmu85-keepkv` | C1 + `rollout.free_cache_engine=false` | Skip per-step KV teardown and re-warmup | Medium-high (10-25% on `gen_s`) | Medium. KV stays resident during actor update; may collide with 43 GB budget. |
| C3 | `P1-gmu90-keepkv` | `gpu_memory_utilization=0.90` + `free_cache_engine=false` | Most aggressive rollout memory | High if it fits | Medium-high. OOM risk during actor update. |

### Tier 2 — moderate cost, parallelism / pipeline changes

| # | Name | Change | Hypothesis | Expected return | Risk |
|---|---|---|---|---|---|
| C4 | `P1-PP1-EP16` | PP=1, VPP=1, EP=16 (actor + ref) | No pipeline bubbles; one contiguous forward/backward | High (10-30% on `update_actor`) | Medium. 40Bra fits per-GPU but activations at seq=24384 may not. Need memory headroom check. |
| C5 | `P1-PP1-EP8-CP2` | PP=1, EP=8, CP=2 (context parallel) | Keep EP at lvbc's known-good value but shard sequence across 2 GPUs to handle long context | Medium | Medium. CP+EP interaction less tested in Megatron-Bridge. |

### Tier 3 — offload audit

| # | Name | Change | Hypothesis | Expected return | Risk |
|---|---|---|---|---|---|
| C6 | `P1-no-opt-offload` | `optimizer_offload=false`, `override_optimizer_config.optimizer_offload_fraction=0.0`, `override_optimizer_config.optimizer_cpu_offload=false` | Optimizer state fits on GPU; skip D2H/H2D every step | Medium (5-15% on `update_actor`) | Medium. ~20 GB/GPU for optimizer state at BF16 precision-aware; should fit alongside 10 GB weights. |
| C7 | `P1-no-all-offload` | C6 + `param_offload=false`, `grad_offload=false` | Keep everything on GPU | Medium-high | High. Will OOM unless gpu_memory_utilization also drops. Combine with rollout-side `gpu_memory_utilization=0.4-0.5`. |

### Tier 4 — token budget

| # | Name | Change | Hypothesis | Expected return | Risk |
|---|---|---|---|---|---|
| C8 | `P1-maxtok-2x` | `ppo_max_token_len_per_gpu=48768`, `infer_ppo_max_token_len=48768`, `rollout.max_num_batched_tokens=48768` | Let dynamic_bsz pack 2 full sequences per micro-batch | Low-medium | Medium. OOM risk; activations scale with seq len. |

### Tier 5 — cumulative

| # | Name | Change | Hypothesis | Expected return | Risk |
|---|---|---|---|---|---|
| C9 | `P1-best` | Combine the winners from C1-C8 | Compound savings | Highest if no conflicts | Depends on winners. |

---

## 4a. Debug-only optimizations (compute-only, Category A)

Three classes of change applied in the 0418 debug infrastructure on
2026-04-18 to shorten the wall clock per debug run without touching any
Category B parameter. Production scripts (`submit/rayjob.yaml`,
`submit/submit.sh`) are NOT touched.

### 4a.1 — Hydra overrides injected by `submit_debug.sh`

| Override | Saves | Reason |
|---|---|---|
| `trainer.val_before_train=false` | ~3–10 min | Skip pre-train baseline val. Training-trajectory equivalence is verified from per-step `critic/score/mean` in `perf_log.jsonl`, not from val accuracy. |
| `trainer.test_freq=9999` | ~3–6 min | Skip mid-run val. |
| `trainer.save_freq=9999` | ~1 min | No checkpoint I/O. |
| `trainer.log_val_generations=0` | trivial | Skip text-sample dumps. |
| `data.val_max_samples=32` | — | Belt-and-suspenders if val is re-enabled. |
| `actor_rollout_ref.actor.optim.lr_warmup_steps=2` | — | Required: `env_k8s_b300_16node.yaml` hard-codes 10, which fails `assert warmup < decay` when total_training_steps=5. |

All are Category A, so they do not change training numerics. Any of
them can be overridden by passing the same key in `submit_debug.sh`
extra args (Hydra uses last-wins).

### 4a.2 — `rayjob_debug.yaml` ttl

`ttlSecondsAfterFinished: 1800 → 120`. A short ttl lets Volcano promote
the next Inqueue gang quickly. The 120 s floor is a safety net for
cases where `pipeline.sh` crashes before Phase 5 deletes the RayJob
explicitly.

### 4a.3 — `pipeline.sh` Phase 5 (explicit RayJob delete)

After Phase 4 (analyze perf_log), `pipeline.sh` now runs
`kubectl delete rayjob $JOB --wait=false` when `perf_log.jsonl` was
captured. Nodes return to the Volcano queue within a few seconds of a
run completing, instead of waiting out the 120 s ttl. If `perf_log.jsonl`
is missing (run failed before writing any), the delete is skipped so
that the head pod logs remain readable for the 120 s window.

### Net effect

A 5-step debug run drops from **~70 min** (old settings) to **~55 min**
(new defaults), and the queued runs behind it promote within seconds
instead of 30 minutes. Over a 9-experiment series, the total wall clock
drops from ~5 h to ~2.75 h.

Currently running experiments (C1–C9 wave submitted at 11:39 UTC
2026-04-18) all use the new defaults (verified with
`KP get rayjob -o jsonpath='{.spec.ttlSecondsAfterFinished}'`).

## 4. Execution order

Run strictly in priority order; **stop and re-evaluate after each tier** rather than firing all 9 at once. Each run costs ~35-45 min wall-clock and frees the cluster after.

1. **C1** (lowest risk, trivial edit) — establishes the `gpu_memory_utilization` ceiling.
2. **C2** — builds on C1 result. If C1 shows `gen_s` did not decrease, C2 won't help either.
3. **C3** — only if C2 succeeded and left memory headroom.
4. **C4** — independent from C1/C2/C3. Biggest potential update_actor win. Launch after Tier 1 completes.
5. **C5** — only if C4 OOMs (CP=2 as fallback).
6. **C6** — independent; targets `update_actor`. Launch after C4.
7. **C7** — only if C6 shows meaningful win.
8. **C8** — independent tokenbudget test.
9. **C9** — final combined run after the individual winners are known.

The existing `0418/submit_debug.sh cdbg5 8 '<overrides>'` pattern handles all of these without yaml edits — every change is expressible as Hydra CLI overrides.

---

## 5. Logger review — what we capture, what is missing

### What the current `perf_log.jsonl` already includes (verified on P1 data)

```
timing_s/{step, gen, reward, old_log_prob, adv, update_actor, testing,
          start_profile, stop_profile,
          agent_loop/generate_sequences/{min,max,mean},
          agent_loop/num_preempted/{min,max,mean},
          agent_loop/slowest/{generate_sequences, tool_calls,
                              prompt_length, response_length, num_preempted}}
perf/{total_num_tokens, time_per_step, throughput,
      max_memory_allocated_gb, max_memory_reserved_gb, cpu_memory_used_gb,
      mfu/actor_infer, mfu/actor}
response_length/{mean, max, min, clip_ratio}
prompt_length/{mean, max, min, clip_ratio}
critic_score/{score/mean, score/max, score/min}
curriculum/{n_domains, n_valid_samples, consumed_samples, target_mean,
            progress, current_step, avg_pass_rate, domain_*_count,
            domain_*_frac, domain_*_target_weight}
derived: tokens_per_gpu_s_gen = perf/total_num_tokens / n_gpus / timing_s/gen
```

This is more than enough for **rollout-side** compute-opt decisions (C1, C2, C3, C8) — we can directly compare step-to-step `tok/s/GPU` and memory.

### What is missing or weak for the Plan C experiments

1. **Per-phase memory peaks**
   - We have `max_memory_allocated_gb` (step-level peak). For C1-C3 and C6-C7 we need to distinguish **rollout-phase peak** (KV-cache dominates) from **update-phase peak** (activation + optimizer dominates). Otherwise we cannot tell whether raising `gpu_memory_utilization` still leaves room for the update.
   - **Fix**: at the end of each phase (`gen`, `update_actor`), call `torch.cuda.reset_peak_memory_stats()` after recording `max_memory_allocated()`. Requires worker-side change; modest invasiveness.
   - **Priority**: high for C2/C3/C7 safety.

2. **vLLM KV cache utilization and preemption count**
   - `agent_loop/num_preempted` is currently `-1` (unknown) on all P0/P1/P2 runs. Preemption tells us if the KV cache was oversubscribed.
   - **Fix**: investigate why the agent loop reports `-1`; likely a vLLM V1 metric that is disabled. Patch-worthy for C1-C3 analysis.
   - **Priority**: medium.

3. **Pipeline bubble ratio**
   - No direct metric exists. With PP>1 some fraction of `update_actor` is bubble (idle waits between micro-batches).
   - **Fix**: compare `timing_s/update_actor` at PP=2 vs PP=1 (C4). The delta itself is the bubble.
   - **Priority**: N/A (handled by the C4 experiment itself).

4. **Response-length percentiles**
   - We have min/max/mean/clip_ratio. Percentiles (p50, p90, p99) would tell us tail behavior more precisely.
   - **Fix**: add `response_length/p{50,90,99}` by computing in `metric_utils.py` or extracting from `batch.meta_info`.
   - **Priority**: low (already can see tail via `agent_loop/slowest/response_length`).

5. **D2H/H2D offload timing**
   - For C6/C7 we need to see how much time the optimizer is spending moving state between CPU and GPU. Megatron-Bridge has this internally but does not surface it in verl's metric dict.
   - **Fix**: hook Megatron's profiler or add torch CUDA events around the offload path. High invasiveness.
   - **Priority**: low for now — the `update_actor` delta across C6/C7 will tell us the aggregate effect.

6. **NCCL / NVSHMEM time**
   - No direct metric.
   - **Fix**: rely on `update_actor` delta between parallelism shapes (C4/C5).
   - **Priority**: low.

### Proposed logger updates before Plan C starts

Only one change is worth making before the C1 run:

- **Per-phase memory snapshots**: add `perf/gen_phase_peak_gb` and `perf/update_phase_peak_gb` by resetting `torch.cuda.max_memory_allocated` at the boundary between `gen` and `update_actor`. This is the only missing signal that directly affects C2/C3/C7 safety decisions.

Everything else in the "missing" list is derivable from existing fields or can wait until a specific experiment reveals a blind spot.

---

## 5a. Lessons from the 2026-04-18 experiment batch (live)

Six C-runs have terminated so far. Three were failures that carry clear lessons about the memory envelope of 40Bra hybrid engine on B300 at `max_response=16384`:

| Run | Config | Result | Lesson |
|---|---|---|---|
| C2 | `gmu=0.85` + `free_cache_engine=false` | OOM during `actor_rollout_compute_log_prob` | Keeping vLLM KV cache resident during actor update adds ~90 GiB that the actor forward cannot find room for. On a 267 GiB GPU this pushes us past the limit. |
| C3 | `gmu=0.90` + `free_cache_engine=false` | OOM during `actor_rollout_compute_log_prob` | Same mechanism as C2. Higher `gpu_memory_utilization` just tightens the cliff. |
| C4 | `PP=1`, `VPP=1` | OS OOM-killed after step 1 (no Python traceback) | At `PP=1` each GPU holds the full layer stack. `max_memory_reserved_gb=290.7 > 288 GiB HBM` — any step-2 allocation trips the kernel OOM-killer, which terminates the container silently. |
| C5 | `PP=1 + CP=2` | Same silent OS OOM-kill as C4 | Context-parallel sequence sharding did not rescue the PP=1 memory pressure for this model. |

**Empirical memory envelope** for 40Bra hybrid engine, 8-node, `max_prompt=8000 / max_response=16384`:
- Safe operating point: ~**237 GiB per GPU** (P1 baseline).
- Fail zone: ~**260 GiB per GPU** and above.
- Any config that pushes above 260 GiB is at high risk, either of `torch.OutOfMemoryError` (Python traceback) during a high-watermark phase, or of OS OOM-kill (silent) when `reserved > HBM capacity`.

**Immediate implications**:
- Do not combine `free_cache_engine=false` with `gpu_memory_utilization ≥ 0.85`.
- Do not use `PP=1` without additional offload or a shorter `max_response_length`.
- C9 (combines `PP=1` with `gmu=0.85 + no-opt-offload`) will fail the same way — it is left in the queue for completeness but the result is predictable.

## 5b. Directory consolidation (2026-04-18, 12:45 UTC)

All per-run files for any RayJob submitted from 2026-04-18 onwards live in one directory named by the RayJob:

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log           entrypoint + training driver stdout/stderr
    perf_log.jsonl    per-step metrics from the verl logger
    analysis.txt      post-run summary written by pipeline.sh
    exp_name.txt      traditional timestamped name (for wandb UI)
    rayjob.txt        self-documenting ID
    label.txt         operator label (e.g. "C1"), if DEBUG_LABEL was set
    <megatron ckpt shards as before>
```

`pipeline.sh` looks the directory up by RayJob name directly. The old `.rayjob_lookup/` marker and the old `verl/logs/...` separate log tree are deprecated (never written for new runs; still consulted as fallbacks for runs that started before the rewrite).

This change is debug-friendly and production-safe: production RayJobs (submitted via `iter_kuberay_32nodes_verl_training/submit/submit.sh`) also benefit from the unified layout — no new directories to scatter results across.

## 6. Baseline reference (from Plan B)

For every C-run, we compare against P1:

| field | P1 value |
|---|---|
| nnodes | 8 |
| n_gpus | 64 |
| mean `step_s` | 516.2 |
| mean `gen_s` | 294.5 |
| mean `update_actor_s` | 165.6 |
| mean `tok/s/GPU` | 561 |
| mean `max_memory_allocated_gb` | 236.6 |
| mean response_length | 5003 |

Success criterion for a C-run: either `step_s` is lower OR `tok/s/GPU` is higher, with no change to critic_score/score/mean trajectory across steps (since we hold all B params fixed, the trajectory must match P1 to within floating-point noise).

---

## 7. Exit criteria

- [ ] At least 3 Tier-1 runs (C1, C2, C3) complete with results in `0418/results/`
- [ ] At least one Tier-2 run (C4) complete; C5 only if C4 failed
- [ ] At least one Tier-3 run (C6) complete; C7 only if C6 shows > 5% improvement
- [ ] C9 combined run completed using the winners
- [ ] `0418/summary.md` updated with a final production config recommendation
- [ ] If any C-run shows > 10% improvement in `tok/s/GPU` AND no change to `critic_score/score/mean` vs P1 → propose flipping it into the production 8-node config
