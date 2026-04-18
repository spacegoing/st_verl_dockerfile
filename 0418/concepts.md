# Concepts you will hit in this project

> Background reading so the plans, scripts, and commit messages make sense.

---

## 1. Apple-to-apple comparison rule

Two training runs are apple-to-apple if and only if they differ only in
Category-A hyperparameters. They may differ in how the same computation is
distributed, allocated, or scheduled, but not in what is computed.

Why it matters: if you want to claim "8-node is 1.16× slower than 16-node
but the training quality is the same," you cannot also change batch size
between the two runs — that would change the gradient math and defeat the
comparison.

Enforcement: `submit_debug.sh` and `submit.sh` keep all Category-B defaults
in the combo yaml. Any Hydra override you pass on the CLI is checked
against the A/B/C split below before accepting.

---

## 2. Category A / B / C hyperparameters

Full list in `plan_c_compute_opt.md` §1. Summary:

### A. Training-irrelevant — safe to change for compute optimization
- Parallelism: `PP`, `VPP`, `TP`, `EP`, `ETP`, `CP`, and node count.
- Memory knobs: `param_offload`, `optimizer_offload`, `grad_offload`,
  `optimizer_offload_fraction`, `gpu_memory_utilization`,
  `max_num_batched_tokens`, `ppo_max_token_len_per_gpu`, `free_cache_engine`.
- Execution: `enforce_eager`, `enable_chunked_prefill`, `use_dynamic_bsz`
  (when kept True throughout the comparison).
- Infra: `nccl_timeout`, NCCL / NVSHMEM / VLLM env vars, reward HTTP
  concurrency.
- Misc: `trainer.test_freq`, `trainer.save_freq`, `data.val_max_samples`,
  `log_val_generations`.

### B. Training-relevant — must NOT change for compute-only comparison
- Batch sizes: `train_batch_size`, `ppo_mini_batch_size`, `ppo_epochs`.
- Sequence lengths: `max_prompt_length`, `max_response_length`.
- Rollout algo: `rollout.n`, `rollout.temperature`, `top_p`, `top_k`.
- Optimization: `lr`, `lr_warmup_steps`, `weight_decay`, `clip_grad`.
- Loss: `policy_loss.loss_mode`, FACPO params (`fapo_delta`, `tau_pos`,
  `tau_neg`), `entropy_coeff`, clip ratios, KL terms.
- Advantages: `algorithm.adv_estimator`, `norm_adv_by_std_in_grpo`,
  `use_kl_in_reward`, `kl_ctrl.*`.
- Data: datasets, curriculum params (`domains`, `domain_balanced`,
  `initial_mean`, `final_mean`, `std`, `total_steps`, `filter_pass_rate_100`).
- Reward: `overlong_buffer_cfg.*`, `max_resp_len`.
- Totals: `total_training_steps`, `total_epochs`.

### C. Gray zone
- `use_dynamic_bsz` when switching between True and False: numerically
  equivalent result but differs in ordering. Keep consistent across a
  comparison.
- `test_freq`, `save_freq`, `val_max_samples`, `val_batch_size`: do not
  affect training but affect which validation numbers you see.

Rule of thumb: if the hyperparameter appears inside a gradient, loss, or
optimizer update, it is in B. If it only affects where bytes live or how
many GPUs run in parallel, it is in A.

---

## 3. Hybrid engine

"Hybrid engine" means the same set of GPUs runs both the actor/Megatron
training and the vLLM rollout inference, alternating within each training
step. This is verl's default (`hybrid_engine: True`).

Per-step flow on one GPU:
1. **Rollout** — vLLM runs generation. Actor params are offloaded to CPU
   (`param_offload`), optimizer + gradients too. vLLM's weights + KV
   cache occupy most of HBM.
2. **Teardown** — vLLM's KV cache is freed (`free_cache_engine=true`).
3. **Update** — Megatron reloads actor params from CPU, runs fwd/bwd on
   the rollout batch, optimizer step.
4. Repeat.

Cost of hybrid vs separate-engine:
- +Simplicity: one cluster, one Ray job.
- +Utilization: every GPU does both work types; no dedicated rollout
  nodes.
- −Per-step overhead: offload/reload of params, vLLM KV warmup.
- −Peak memory pressure: the transition points (KV-to-actor, actor-to-KV)
  are where OOM usually happens.

Our measured ceiling for 40Bra hybrid engine: ~237 GiB per GPU (P1
baseline). At 260+ GiB per GPU, OS OOM-kill becomes a real risk.

---

## 4. Gang scheduling (KubeRay + Volcano)

Gang = all-or-nothing placement. For a multi-pod workload, gang scheduling
means either every pod binds to a node at once, or no pod binds at all.
Prevents partial-placement deadlocks where N jobs each hold some but not
all of their nodes.

On this cluster:
- kuberay-operator has `--batch-scheduler=volcano`. When a RayJob is
  created, the operator also creates a Volcano PodGroup with
  `minMember = 1 + workerReplicas`.
- Volcano's scheduler has the `gang` plugin enabled. It holds the
  PodGroup `Inqueue` until `minMember` nodes satisfying the resource
  request are simultaneously free, then binds all pods atomically.

Observable states:
- `PodGroup.status.phase == Inqueue` — waiting for capacity.
- `PodGroup.status.phase == Running` — all pods placed.
- `PodGroup.status.conditions[?type=Scheduled].status == True` — gang
  admitted.

Verification: `0418/verify_gang.sh <rayjob_name>` runs six checks. Full
background in `0418/plan_a_kuberay_volcano.md`.

---

## 5. MoE sharding axes on 40Bra

40Bra is a DeepSeek-V3-family MoE (~40B parameters). Megatron shards the
model across four parallelism axes. Each axis has a size ≥ 1, and their
product must equal the total GPU count (usually `nnodes × 8`).

| Axis | Meaning | Our default |
|---|---|---|
| `TP` (tensor_model_parallel_size) | Split each layer's matrix ops across TP GPUs. | 1 |
| `PP` (pipeline_model_parallel_size) | Split layers into PP sequential stages. | 2 |
| `VPP` (virtual_pipeline_model_parallel_size) | Interleave pipeline stages to hide bubbles. | 2 |
| `EP` (expert_model_parallel_size) | Distribute experts across EP GPUs. | 8 |
| `ETP` (expert_tensor_parallel_size) | TP inside each expert. | 1 |
| `CP` (context_parallel_size) | Shard long sequences across CP GPUs. | 1 |
| `DP` (derived) | Data parallel replicas. | 8 (at 128 GPUs) |

Math: `DP × PP × TP × EP × CP = total_gpus`. At 16 nodes × 8 GPU = 128:
`DP=8, PP=2, TP=1, EP=8, CP=1` → `8×2×1×8×1 = 128`. ✓

Each DP replica spans `PP × TP × EP × CP = 2 × 1 × 8 × 1 = 16` GPUs.

Changing `PP` or `EP` changes the physical layout but produces
numerically equivalent gradients (modulo reduction ordering). That is why
they are in Category A.

---

## 6. Why `PP=1` is dangerous on 40Bra

Empirical result from Plan C (C4, C5): `PP=1` on 40Bra at
`max_response=16384` OOMs deterministically. Why:

- At `PP=2`, each GPU holds half the model layers. Per-GPU weight memory
  is ~50 GiB.
- At `PP=1`, each GPU holds ALL layers. Per-GPU weight memory is ~100 GiB.
- Activations for `max_seq = 8000 + 16384 ≈ 24K tokens` at the full layer
  stack push `max_memory_reserved_gb` past 288 GiB (HBM capacity).
- Once reserved > capacity, any subsequent allocation trips the OS
  OOM-killer. The killer terminates the container without giving Python
  a chance to raise. You get no traceback — the log just stops.

To use `PP=1`, you must also either halve `max_response_length`
(Category B — breaks apple-to-apple) or enable aggressive offload.
Neither is compatible with the current Plan C goal.

---

## 7. Why `free_cache_engine=false` is dangerous at high `gpu_memory_utilization`

`gpu_memory_utilization` tells vLLM what fraction of HBM to reserve for
its weights + KV cache. At 0.85 on a 288 GiB B300: ~245 GiB.

With `free_cache_engine=true` (our default), vLLM frees its KV cache
between steps so the actor update has the full remaining memory to work
with.

With `free_cache_engine=false`, KV stays resident. The actor update
competes with vLLM for the remaining memory. Empirical result: OOM during
`actor_rollout_compute_log_prob` at gmu ≥ 0.85.

If you want keepkv, drop gmu to ≤ 0.5 AND enable more offload. The
trade-off against the per-step KV warmup savings must be measured, not
assumed.

---

## 8. The Volcano queue is your friend

The cluster has 31 compute nodes. Each production run needs 16 nodes; each
debug run needs 8. If you submit more RayJobs than can fit, Volcano
queues them — it does NOT fail or reject them.

Practical consequence: `queue_all.sh` submits all 9 C-runs at once. Three
run concurrently (24 nodes used); six stay `Inqueue`. As each running run
finishes, the next queued gang gets admitted atomically. Total throughput
is the same as if you ran them serially, but you do not have to wait
between runs.

Do not worry about submitting "too many". Worry about submitting the
wrong ones.

---

## 9. Proxy gotcha

The system proxy (`http://jdtcom:...@10.119.176.202:3128`) is needed for
pip / curl to reach the internet, but it BLOCKS the cluster's k8s API
server at `180.184.249.201`. Symptom: `kubectl` returns
`dial tcp: lookup ... : no such host` or `context deadline exceeded`.

Workaround used everywhere in this tree:
```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl ...
```

Save yourself the typing with a shell function:
```bash
KP() { HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl "$@"; }
```

Some scripts (`submit.sh`, `submit_debug.sh`, `pipeline.sh`) do this
unset internally, so they work regardless of your shell's proxy state.

---

## 10. Hydra config groups and `@package _global_`

The env profiles under `verl/my_scripts/k8s/config/env/` are selected via a
Hydra config group:

```yaml
# 40bra_16node_sd.yaml
defaults:
  - env: k8s_b300_16node     # picks env/k8s_b300_16node.yaml
```

Hydra's default behavior for a file under a config group is to place its
contents **under the group name** in the composed config. So without further
action, `env/k8s_b300_16node.yaml` would contribute
`cfg.env.actor_rollout_ref.*` instead of `cfg.actor_rollout_ref.*`. The real
actor config would not be overridden.

To make the file's content merge at the root of the composed config, the
first line of each env yaml must be:

```yaml
# @package _global_
```

This directive tells Hydra to treat the file as a root-level override, not a
subtree. Without it, `actor_rollout_ref.model.path` in the env yaml does
nothing; with it, the env yaml fully controls the actor / rollout / data
config for the run.

**Rule of thumb**: any yaml under a Hydra config group that is meant to
override global config must start with `# @package _global_`. This is what
lets `NNODES=8 ./submit.sh c351` correctly switch to `k8s_b300_8node_c10`'s
no-offload settings.

This subtlety bit us at 15:30 UTC 2026-04-18 on the first test of the new
`NNODES` interface — see `dev_notes.md` for the failure signature.

## 11. Why everything lives on PFS under one directory per RayJob

Old layout had four different locations with three different IDs per run.
Operator pain: to read one run, you jumped between `verl/logs/...`,
`verl/ckpts/...`, `0418/results/...`, and `/tmp/...`.

New layout (since 2026-04-18 12:45 UTC): one directory on PFS keyed by
the RayJob name, containing every per-run artifact:

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log, perf_log.jsonl, analysis.txt,
    exp_name.txt, rayjob.txt, label.txt,
    <megatron ckpt shards>
```

To inspect a run you now need one command:
```bash
ls /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/
```

This applies to both production and debug. `pipeline.sh` also caches
flat copies under `0418/results/<label>_*` for cross-run comparison
tools, but those are disposable — the canonical source is the per-RayJob
directory.
