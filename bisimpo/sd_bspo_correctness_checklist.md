# BSPO Single-Domain 40Bra — Correctness Checklist

Four-section checklist for a BSPO sd training job on 40Bra:

1. **Quick start** — full manual submission path with every file touched.
2. **KubeRay correctness** — k8s side.
3. **verl training correctness** — training loop, data, config composition.
4. **BSPO algo correctness** — loss math, variant dispatch.

All paths are repo-relative. "Local" = host PFS
(`/mnt/public/lichang93/st_verl_dockerfile/`). "Container" =
`/root/myCodeLab/host/` inside the kuberay pod.

### Two parallel submission pipelines

This dev cluster (B300) submits via a **split-repo** pipeline. The
consolidated `verl/my_scripts/bspo_scripts/` copy only exists on the
`verl@nemo_bspo_md` branch and is intended for the B200 cluster. Both
pipelines have byte-identical logic — use whichever matches the branch
you have checked out.

| role                 | this cluster (dev, `verl@nemo_bspo_sd_ablation`) | B200 (`verl@nemo_bspo_md`) |
|----------------------|--------------------------------------------------|-----------------------------|
| combo dispatcher     | `bisimpo/submit_bspo.sh`                         | `verl/my_scripts/bspo_scripts/submit_bspo.sh` |
| k8s dispatcher       | `iter_kuberay_32nodes_verl_training/submit/submit.sh` | `verl/my_scripts/bspo_scripts/submit.sh` |
| RayJob template      | `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml` | `verl/my_scripts/bspo_scripts/rayjob.yaml` |

Only the location differs; the files are verbatim copies of each other.
The checklist below uses the **dev-cluster** paths. Swap them for the
B200 paths if reviewing that clone.

---

## 1. Quick start — end-to-end manual submission

### Example command

```bash
# dev cluster (this repo layout):
cd /mnt/public/lichang93/st_verl_dockerfile
NNODES=4 bisimpo/submit_bspo.sh cbsp401 \
    'actor_rollout_ref.actor.bspo_delta=3.0e-4 actor_rollout_ref.actor.bspo_lambda_tj=1.0e-2'

# B200 cluster (after `verl` checked out at nemo_bspo_md):
# NNODES=16 verl/my_scripts/bspo_scripts/submit_bspo.sh cbsp401 '...'
```

### Call stack + file list (top-down)

```
[USER SHELL]
  │
  ▼
bisimpo/submit_bspo.sh
  - combo_id → {VAR, DELTA, LAMBDA, LR_WARMUP} case table
  - Builds BSPO_OVR + ABLATE_OVR hydra overrides
  - NNODES default: 4
  - exec $HERE/../iter_kuberay_32nodes_verl_training/submit/submit.sh cbsp401 "<overrides>"
  │
  ▼
iter_kuberay_32nodes_verl_training/submit/submit.sh
  - unset HTTPS_PROXY / export NO_PROXY=<k8s api ip>
  - NNODES case: 16→k8s_b300_16node, 8→k8s_b300_8node_c10,
    4→k8s_b300_4node_c11, 2→k8s_b300_2node_debug
  - WORKER_REPLICAS = NNODES - 1
  - EXTRA_OVERRIDES = "env=${ENV_NAME} <overrides>"
  - combo_id regex check: ^c[a-z0-9-]+$
  - envsubst on rayjob.yaml with whitelist
    {COMBO_ID, NNODES, WORKER_REPLICAS, EXTRA_OVERRIDES}
  - kubectl create -f -
  │
  ▼
iter_kuberay_32nodes_verl_training/submit/rayjob.yaml
  - apiVersion: ray.io/v1, kind: RayJob
  - generateName: bra40-sd-${COMBO_ID}-
  - spec.entrypoint: bash .../run_40bra_k8s_16node_single_domain.sh
                        ${COMBO_ID} ${EXTRA_OVERRIDES}
  - rayClusterSpec.headGroupSpec.template.spec.containers:
      image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
      imagePullSecrets: aliyunsecret
      env: NCCL/NVSHMEM/VLLM/PYTHONPYCACHEPREFIX
      resources: nvidia.com/gpu: 8, rdma-training/roce: 1, cpu: 176, mem: 1920Gi
      volumeMounts: afs-pvc → /root/myCodeLab/host
  - workerGroupSpecs: replicas: ${WORKER_REPLICAS}
  │
  ▼ [kubectl → kuberay-operator → Volcano gang scheduler]
  │
  ▼  (pods up, ray-head runs its entrypoint)
verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh
  - RAY_DATA_HOME=/root/myCodeLab/host/downloads
  - MODEL_PATH=$RAY_DATA_HOME/models/40Bra
  - exp_name = 40bra_k8s_16node_sd_${COMBO_ID}_${TIMESTAMP}_${COMMIT_ID}
  - CKPTS_DIR = /root/myCodeLab/host/verl/ckpts/40bra_k8s_single_domain/${RAYJOB_NAME}/
  - bash verl/my_scripts/gym/start_gym_uv.sh        (start Gym on head)
  - python3 verl/my_scripts/gym/start_gym_all_nodes.py  (start Gym on workers via Ray)
  - export WANDB_* + TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  - exec &> >(tee -a "${RUN_DIR}/run.log")
  - RAY_ADDRESS=auto ray job submit \
      --runtime-env=${RUNTIME_ENV} \
      -- python3 -m verl.trainer.main_ppo \
          --config-path=verl/my_scripts/k8s/config \
          --config-name=40bra_16node_sd \
          combo_id=${COMBO_ID} trainer.experiment_name=${exp_name} \
          trainer.default_local_dir=${CKPTS_DIR} "${@:2}"
  │
  ▼
verl/my_scripts/gym/start_gym_uv.sh
verl/my_scripts/gym/start_gym_all_nodes.py
  - Starts isolated Ray on port 6380 (no conflict with verl's 6379)
  - Launches 7 Gym resource servers on ports 20001–20007
  - localhost URLs consumed by reward_manager
  │
verl/my_scripts/k8s/my_deepep_env_k8s.yaml
  - Ray runtime env: NVSHMEM_BOOTSTRAP, NVSHMEM_IB_ENABLE_IBGDA,
    OTEL_SDK_DISABLED, VLLM_USE_V1, etc.
  │
  ▼  (ray job submit launches main_ppo)
verl/verl/trainer/main_ppo.py
  - Hydra @main decorator: config_path, config_name from CLI
  │
  ▼  (Hydra composes the config)
verl/my_scripts/k8s/config/40bra_16node_sd.yaml         [base]
  defaults:
    - ppo_megatron_trainer         [verl/trainer/config/...]
    - env: k8s_b300_16node         [overridable at CLI]
    - combo_40Bra@combo_bank       [all combo defs]
    - _self_
verl/my_scripts/k8s/config/env/k8s_b300_4node_c11.yaml  [when env=k8s_b300_4node_c11]
  - actor_rollout_ref.model.path, data.train_files, data.val_files
  - Megatron PP/EP/TP, no offload, gmu=0.4, max_response_length=16384
verl/my_scripts/k8s/config/combo_40Bra.yaml             [all cbspXXX definitions]
  - cbsp401: {fapo_delta, ppo_epochs, total_training_steps, domains, …}
verl/verl/trainer/config/ppo_megatron_trainer.yaml      [verl defaults]
  - defaults for actor, rollout, ref, reward_model, trainer
  │
  ▼  (PPO training starts)
verl/verl/trainer/ppo/ray_trainer.py
  - Dataset + val dataloader construction with curriculum sampler
  - For each step: rollout → reward → advantages → update_actor → val
  - uid injection at line 1429 (one uuid per prompt)
  │
verl/verl/utils/dataset/rl_dataset.py               [row dict → tensors]
verl/verl/experimental/dataset/curriculum_sampler.py [Gaussian linear ramp]
verl/verl/workers/reward_manager/nemogym_server.py  [HTTP → local Gym]
  │
  ▼  (each update_actor: Ray dispatch to Megatron workers)
verl/verl/workers/megatron_workers.py
  - update_actor: make_minibatch_iterator → update_policy
  │
verl/verl/workers/actor/megatron_actor.py
  - forward_backward_batch:
      * broadcast mini_batch.batch across PP group
      * build micro_batches (split or rearrange_micro_batches)
      * forward_step on each micro_batch
      * loss_func(output, data, meta_info) on last PP rank
  │
  ▼
verl/verl/trainer/ppo/core_algos.py
  - get_policy_loss_fn(loss_mode="bspo") → compute_policy_loss_bspo
  - Returns (pg_loss, pg_metrics)
  │
  ▼ loss.backward() → optimizer.step() → grad all-reduce across DP
  │
  ▼  (metrics logged to wandb)
verl/wandb_my_dirs/wandb/offline-run-<timestamp>-<run_id>/
  │
  ▼  (checkpoints written on save_freq)
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
```

### Full file inventory touched by one submission

| layer            | file (dev cluster)                                                   |
|------------------|----------------------------------------------------------------------|
| shell wrapper    | `bisimpo/submit_bspo.sh`                                             |
| kuberay submit   | `iter_kuberay_32nodes_verl_training/submit/submit.sh`                |
| rayjob template  | `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml`              |
| pod entrypoint   | `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh`          |
| Gym startup      | `verl/my_scripts/gym/start_gym_uv.sh`                                |
|                  | `verl/my_scripts/gym/start_gym_all_nodes.py`                         |
| Ray runtime env  | `verl/my_scripts/k8s/my_deepep_env_k8s.yaml`                         |
| Hydra base       | `verl/my_scripts/k8s/config/40bra_16node_sd.yaml`                    |
| Hydra env        | `verl/my_scripts/k8s/config/env/k8s_b300_{16,8,4,2}node*.yaml`       |
| Hydra combos     | `verl/my_scripts/k8s/config/combo_40Bra.yaml`                        |
| verl defaults    | `verl/verl/trainer/config/ppo_megatron_trainer.yaml`                 |
|                  | `verl/verl/trainer/config/actor/actor.yaml`                          |
| main entry       | `verl/verl/trainer/main_ppo.py`                                      |
| trainer loop     | `verl/verl/trainer/ppo/ray_trainer.py`                               |
| dataset          | `verl/verl/utils/dataset/rl_dataset.py`                              |
| curriculum       | `verl/verl/experimental/dataset/curriculum_sampler.py`               |
| reward manager   | `verl/verl/workers/reward_manager/nemogym_server.py`                 |
| worker dispatch  | `verl/verl/workers/megatron_workers.py`                              |
| actor forward    | `verl/verl/workers/actor/megatron_actor.py`                          |
| actor config     | `verl/verl/workers/config/actor.py`                                  |
| BSPO loss        | `verl/verl/trainer/ppo/core_algos.py`                                |
| BSPO spec ref    | `bisimpo/bspo_algorithm.tex` (Single-domain BSPO §)                  |

---

## 2. KubeRay correctness checklist

Everything here sits between `submit_bspo.sh` and the first Python line
of `main_ppo`. If the RayJob never gets its pods to Running with
Ray+GPU ready, verl never starts.

### 2.1 `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml`

- [ ] **Image**: `image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` — registry reachable, tag exists.
- [ ] **Pull secret**: `imagePullSecrets[].name: aliyunsecret` — secret exists in the namespace.
- [ ] **GPU resource**: `nvidia.com/gpu: '8'` — matches the device-plugin exposed resource name.
- [ ] **RDMA resource**: `rdma-training/roce: '1'` — matches what the RoCE device plugin advertises (different clusters may use `rdma/hca` or `nvidia.com/roce`).
- [ ] **Volume**: `persistentVolumeClaim.claimName: pvc-jdwzpnv` — PVC bound, PFS readable from pod.
- [ ] **subPath**: `lichang93/st_verl_dockerfile` → `/root/myCodeLab/host` — pod sees the repo at the hardcoded container path.
- [ ] **subPath**: `lichang93/downloads` → `/root/myCodeLab/host/downloads` — model + dataset visible at the path the entrypoint expects.
- [ ] **Shm**: `emptyDir.sizeLimit: 60Gi` — large enough for torch shared tensors.
- [ ] **NCCL envs**: `NCCL_IB_HCA` lists the correct 8 GPU-affine NICs (`mlx5_10…mlx5_17` on B300). GID index matches RoCEv2 config (5 here).
- [ ] **NVSHMEM envs**: `NVSHMEM_HCA_LIST`, `NVSHMEM_IBGDA_NIC_HANDLER=gpu` — keep `gpu` unless DCT init fails.
- [ ] **VLLM backend**: `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` required on B300 (sm_103). Omit on newer SMs.
- [ ] **envsubst whitelist intact**: `${COMBO_ID}`, `${NNODES}`, `${WORKER_REPLICAS}` appear literally in the template; `submit.sh` must whitelist exactly these.

### 2.2 `iter_kuberay_32nodes_verl_training/submit/submit.sh`

- [ ] `NO_PROXY` set to the k8s API IP so `kubectl` bypasses the proxy.
- [ ] `NNODES` case statement maps to an existing env yaml under `verl/my_scripts/k8s/config/env/`. An unrecognised NNODES exits 2.
- [ ] `WORKER_REPLICAS = NNODES − 1` (head takes 1, workers take the rest).
- [ ] Combo-id regex `^c[a-z0-9-]+$` allows hyphens (needed for smoke combos like `cbdg-md-v1-smoke`).
- [ ] `envsubst` whitelist matches exactly the placeholders in `rayjob.yaml`.
- [ ] `kubectl create -f -` (not `apply`) so each submission gets a unique `generateName` suffix.

### 2.3 `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh` (pod entrypoint)

- [ ] `RAY_DATA_HOME` defaults resolve to a real PFS path inside the pod.
- [ ] `RAYJOB_NAME` derivation from hostname strips the kuberay suffix correctly.
- [ ] Gym startup sequence: head first (`start_gym_uv.sh` direct bash), then workers via Ray tasks (`start_gym_all_nodes.py`). If head Gym fails the script must exit before `ray job submit`.
- [ ] `WANDB_MODE=offline` exported (pod has no outbound to wandb.ai).
- [ ] `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` for clean crash on NCCL timeout.
- [ ] `exec &> >(tee -a ${RUN_DIR}/run.log)` redirects both stdout and stderr to the run's log on PFS.
- [ ] `RAY_ADDRESS=auto` — connects to the Ray cluster started by the pod.
- [ ] `--runtime-env` points at `my_deepep_env_k8s.yaml` (not the baremetal one).
- [ ] `"${@:2}"` passes CLI overrides through verbatim.

### 2.4 `verl/my_scripts/gym/start_gym_uv.sh` + `start_gym_all_nodes.py`

- [ ] 7 resource servers start on ports 20001–20007; health-check endpoint returns 200 before the script exits.
- [ ] Uses an isolated Ray cluster (separate `--temp-dir`, port 6380) so it doesn't collide with verl's Ray (port 6379).
- [ ] `start_gym_all_nodes.py` uses Ray `NodeAffinitySchedulingStrategy` so each worker gets exactly one Gym startup task.

### 2.5 `verl/my_scripts/k8s/my_deepep_env_k8s.yaml`

- [ ] No `working_dir` key (verl is an editable install mounted via PFS).
- [ ] `NVSHMEM_BOOTSTRAP: IB` — UID socket doesn't work cross-pod.
- [ ] `NVSHMEM_IBGDA_NIC_HANDLER: cpu_host_memory` — DCT-free path, required on this fabric.
- [ ] `RAY_OVERRIDE_JOB_RUNTIME_ENV: 1`, `VLLM_USE_V1: 1`.

### 2.6 Post-submit validation commands

```bash
# gang scheduled
kubectl get rayjob -l combo_id=cbsp401
kubectl get pods  -l ray.io/cluster=<rayClusterName>
# Ray cluster healthy
kubectl exec <head-pod> -- ray status
# all 7 Gym ports up
kubectl exec <head-pod> -- bash my_scripts/gym/start_gym_uv.sh --status
```

---

## 3. verl training correctness checklist

Everything from the first Python line up to the policy-loss call.
Goal: the training loop sees the right data, right config, right
parallelism, right curriculum, and logs usable metrics.

### 3.1 Hydra config composition

- [ ] `verl/my_scripts/k8s/config/40bra_16node_sd.yaml` — base:
  - `defaults:` lists `ppo_megatron_trainer`, `env: k8s_b300_16node`, `combo_40Bra@combo_bank`, `_self_`.
  - `hydra.searchpath: [pkg://verl.trainer.config]` so `ppo_megatron_trainer` resolves.
  - `combo_id` default set; `trainer.experiment_name` and `trainer.default_local_dir` MUST be overridden on CLI (checked by entrypoint).
  - Interpolations `${combo_bank.${combo_id}.*}` resolve for the submitted combo.
- [ ] `verl/my_scripts/k8s/config/env/k8s_b300_4node_c11.yaml` (or matching):
  - `# @package _global_` header (merges at root, not under `cfg.env.*`).
  - `actor_rollout_ref.model.path` → exists in PFS.
  - `data.train_files`, `data.val_files` → exist in PFS.
  - Megatron shard plan makes sense: `n_gpus_per_node × nnodes = PP × VPP(dropped in PP size) × EP × TP × DP`.
  - `train_batch_size` divisible by `ppo_mini_batch_size` (and both divisible by DP×rollout.n).
  - `max_response_length`, `max_prompt_length` consistent with dataset.
- [ ] `verl/my_scripts/k8s/config/combo_40Bra.yaml`:
  - Combo entry defines exactly the fields interpolated from the base file (`fapo_delta`, `ppo_epochs`, `total_training_steps`, `domains`, `curriculum_total_steps`, `save_freq`).
  - `total_training_steps > lr_warmup_steps` (Megatron asserts this).
  - `domains` field is a valid token for the curriculum sampler (`"nemogym_math"`, `""` = auto, or `name=src1+src2:weight` form).
- [ ] `verl/verl/trainer/config/ppo_megatron_trainer.yaml` — not usually edited; verify its defaults haven't drifted.

### 3.2 Dataset + curriculum

- [ ] `verl/verl/utils/dataset/rl_dataset.py` — row_dict contains `prompt`, `data_source`, `extra_info`. `extra_info` is either dict-valued or JSON string.
- [ ] `verl/verl/experimental/dataset/curriculum_sampler.py`:
  - `pass_rate_key: jd_pass_rate` for sd (nemogym_math), `nemo_pass_rate` for md (nemogym_blend).
  - `domain_key: data_source`.
  - `filter_pass_rate_100: true` drops trivial samples.
  - `initial_mean=0.8 → final_mean=0.2` over `curriculum_total_steps`.
  - `domain_balanced: false` for sd single-domain runs.

### 3.3 Rollout + reward

- [ ] `verl/my_scripts/k8s/config/40bra_16node_sd.yaml` `reward_model`:
  - `reward_manager: nemogym_server`.
  - `server_urls` maps each `data_source` to a localhost Gym port (20001–20006).
  - For sd math add `beyondaime`, `aime2025`, `olympiadbench` → `http://localhost:20006` (all dispatched to math server for eval).
- [ ] `verl/verl/workers/reward_manager/nemogym_server.py`:
  - URL dispatch by `data_source` string.
  - `overlong_buffer_cfg.enable` matches the env file's `max_resp_len`.

### 3.4 Trainer orchestration

- [ ] `verl/verl/trainer/ppo/ray_trainer.py`:
  - Line ~1429: `batch.non_tensor_batch["uid"] = np.array([uuid.uuid4() str for _ in batch])` — one uid per PROMPT, before `gen_batch.repeat(n, interleave=True)` on line ~1437. All n responses of a prompt share a uid (group semantics).
  - `compute_advantage` uses GRPO normalisation.
  - `_update_actor` and `_update_critic` dispatch paths.
  - Val cadence controlled by `trainer.test_freq`; baseline val at step 0 if `trainer.val_before_train=true`.
- [ ] `verl/verl/workers/megatron_workers.py:update_actor`:
  - `dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")` — DP-shards the batch, replicates within PP/EP/TP.
  - `make_minibatch_iterator` called before `update_policy`.

### 3.5 Actor forward + loss dispatch

- [ ] `verl/verl/workers/actor/megatron_actor.py`:
  - `make_minibatch_iterator` select keys include `advantages`, `old_log_probs`, `response_mask`, `input_ids`, `attention_mask`, `position_ids`, `responses`.
  - `forward_backward_batch`:
    * `broadcast_dict_tensor(mini_batch.batch, src=last_pp_rank, group=pp_group)` broadcasts batch tensors across PP.
    * Micro-batches built with `rearrange_micro_batches` (dynamic bsz) or `.split(micro_batch_size)` (static).
  - `loss_func`:
    * `loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")` — for bspo, must be `"bspo"`.
    * `policy_loss_fn = get_policy_loss_fn(loss_mode)` returns `compute_policy_loss_bspo`.
    * Passes kwargs: `old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights`.
    * Result: `pg_loss, pg_metrics` merged into stats.
- [ ] `verl/verl/workers/config/actor.py` — `ActorConfig` has `bspo_variant`, `bspo_delta`, `bspo_lambda_tj` fields with defaults.

### 3.6 Metrics + checkpointing

- [ ] `trainer.save_freq` vs `total_training_steps` — save a final ckpt or set `save_freq=9999` to skip.
- [ ] `trainer.logger: [console, wandb]` and `project_name: 40bra_k8s_single_domain`.
- [ ] wandb offline dir on PFS: `verl/wandb_my_dirs/wandb/offline-run-*`. No `working_dir` in Ray runtime env, so code reloads from PFS editable install on every worker restart.
- [ ] `actor/pg_loss`, `actor/grad_norm`, `actor/ppo_kl`, `actor/bspo_*` all appear in wandb (see 4.4).
- [ ] `val-core/{beyondaime,aime2025,olympiadbench}/acc/mean@1` appear each val checkpoint.

---

## 4. BSPO algorithm correctness checklist

### 4.1 Math reference

- [ ] `bisimpo/bspo_algorithm.tex` §Single-domain BSPO (lines 20–71) is the spec. Read alongside the code.

### 4.2 `verl/verl/trainer/ppo/core_algos.py` — `compute_policy_loss_bspo`

Currently at lines ~1741–1858. Check each step matches the spec:

- [ ] **Log ratio clamp**: `log_ratio = clamp(log_prob - old_log_prob, min=-20, max=20)` — prevents NaN from exp overflow.
- [ ] **Per-trajectory stats** (arithmetic form per spec §Ratio statistics):
  ```python
  dev_pos = clamp(ratio - 1, min=0) * response_mask
  dev_neg = clamp(1 - ratio, min=0) * response_mask
  s_tau_pos = dev_pos.sum(dim=-1) / seq_lens      # (B,)
  s_tau_neg = dev_neg.sum(dim=-1) / seq_lens
  s_tau = s_tau_pos - s_tau_neg
  ```
  `seq_lens` is `response_mask.sum(dim=-1).clamp(min=1)` (divide-by-zero guard).
- [ ] **Per-trajectory advantage**: `A_tau = (advantages * response_mask).sum(-1) / seq_lens`.
  Under GRPO, `advantages` is constant along t within a trajectory, so this is just that constant.
- [ ] **Clipping operators**:
  ```python
  delta_t     = tensor(bspo_delta)
  s_min       = minimum(s_pos, s_neg)
  delta_reg   = clamp(s_pos, max=delta_t) - clamp(s_neg, max=delta_t)   # (B,) matches spec Δ^(reg)
  sign_s      = sign(s_tau).detach()                                    # Q8: MUST be .detach() for V4
  delta_w     = sign_s * clamp(s_min, max=delta_t)                      # (B,) matches spec Δ^(W)
  penalty     = clamp(s_min / delta_t - 1, min=0)                       # (B,) matches spec pen_τ, ≥ 0
  ```
- [ ] **Variant dispatch**: exactly 3 branches, matching spec §Per-trajectory objective:
  ```python
  if variant == "simplest":        per_traj = delta_reg * A_tau                             # V1
  elif variant == "w_penalty":     per_traj = delta_w * A_tau - lam * penalty               # V4
  else:  # "w_penalty_only"        per_traj = s_tau * A_tau - lam * penalty                 # V5
  ```
  Assertion: `variant in {"simplest", "w_penalty", "w_penalty_only"}`.
- [ ] **Aggregation** (spec §Aggregated loss):
  ```python
  per_token_obj = per_traj.unsqueeze(-1).expand_as(advantages)
  pg_losses     = -per_token_obj                                        # negate for gradient descent
  pg_loss       = agg_loss(pg_losses, response_mask,
                            loss_agg_mode="seq-mean-token-mean",
                            **config.global_batch_info)
  ```
  `loss_agg_mode` is **hardcoded** to `"seq-mean-token-mean"` per Q7 — not the value read from config.
- [ ] **Metrics emitted** — 14 keys under `actor/*`:
  - `pg_clipfrac` = `((s_pos > δ) | (s_neg > δ)).float().mean()`
  - `pg_clipfrac_lower = 0` (not used in bspo)
  - `ppo_kl = masked_mean(-log_ratio, response_mask)`
  - `bspo_s_{pos,neg,tau,tau_abs}_mean`
  - `bspo_delta_{reg,w}_mean`
  - `bspo_clip_{pos,neg}_frac`
  - `bspo_penalty_{active_frac,mean}`
  - `bspo_per_traj_{mean,abs_mean}`

### 4.3 `verl/verl/workers/config/actor.py`

- [ ] `ActorConfig` fields present:
  ```python
  bspo_variant:      str    = "simplest"      # simplest | w_penalty | w_penalty_only
  bspo_delta:        float  = 3.0e-4
  bspo_lambda_tj:    float  = 1.0e-3
  ```
- [ ] No typos in names that would break interpolation from combo yaml.

### 4.4 `verl/verl/trainer/config/actor/actor.yaml`

- [ ] Defaults match the dataclass — Hydra should never create a default-missing error at compose time.

### 4.5 Wiring check — does `bspo` get dispatched?

- [ ] In the submitted config, `actor_rollout_ref.actor.policy_loss.loss_mode == "bspo"` (set by `submit_bspo.sh`).
- [ ] In `megatron_actor.py:506`: `loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")`. If this returns `"vanilla"` the run is NOT bspo.
- [ ] In `core_algos.py:POLICY_LOSS_REGISTRY`: `"bspo"` maps to `compute_policy_loss_bspo`. Check at runtime:
  ```python
  from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY
  assert "bspo" in POLICY_LOSS_REGISTRY
  ```
- [ ] At step 1 wandb should show `actor/bspo_*` keys. If only `actor/pg_loss` without `bspo_*`, the wrong loss ran.

### 4.6 Numerical sanity at step 1

On the first training step (on-policy under `ppo_epochs=1` or GRPO):

- [ ] `actor/bspo_s_tau_abs_mean` is small but non-zero (≈ 1e-3 to 1e-2 — due to vLLM vs Megatron log-prob gap, not exactly 0).
- [ ] `actor/bspo_clip_pos_frac` and `actor/bspo_clip_neg_frac` each ≈ 0.5 after a few steps (both sides of the clip start triggering as s_τ grows).
- [ ] `actor/grad_norm` finite (no NaN / inf). For V1 early: < 1; for V4/V5 early: can be 100+ (clip_grad=1.0 catches it).
- [ ] `actor/pg_loss` has the right sign: negative means the policy is being pushed toward higher-advantage trajectories (correct for minimisation of `-per_token_obj`).

### 4.7 End-to-end sanity

- [ ] `val-core/beyondaime/acc/mean@1` at step 0 ≈ 0.45–0.55 (base 40Bra).
- [ ] `val-core/aime2025/acc/mean@1` at step 0 ≈ 0.63–0.80.
- [ ] `val-core/olympiadbench/acc/mean@1` at step 0 ≈ 0.65–0.74.
- [ ] By step 80 (V1) or step 120 (V5 winner), val trajectory matches the pattern in `bisimpo/sd_ablation_report.tex`:
  - V1 simplest: peak at step 50–80 (e.g.\ cbsp203 AIME = 0.83), then crash at step 90.
  - V4 w_penalty: monotone decay across 120 steps.
  - V5 w_penalty_only at (δ=3e-4, λ=1e-2): flat at baseline through step 120.

If your run deviates materially from this pattern at the same `(δ, λ)`
point, something in sections 2 or 3 is wrong — the loss itself is
consistent across the cbsp401/cbsp406 paired replicates.
