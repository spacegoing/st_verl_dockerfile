# QUICKSTART — st_verl_dockerfile

> Operator quick-reference for building images, running baremetal training, and launching k8s RayJobs.
> For deep-dive context see the **File Map** at the bottom of this document.

---

## 0. Cluster Overview

| Host | IP | Role |
|---|---|---|
| b32 | 10.12.11.6 | Ray head, where Claude Code runs |
| b31 | 10.12.11.5 | Ray worker (2-node only) |

**Two images** (registry: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl`):

| Tag | Dockerfile | Rebuild when | Build time |
|---|---|---|---|
| `ncr2602_vllm012.base` | `Dockerfile.base` | vLLM / CUDA ext / system pkg change | ~30 min |
| `ncr2602_vllm012.dev` | `Dockerfile.ncr.26.02.mydev` | verl / Gym code change | ~10-15 min |

**Container layout**: two containers are mutually exclusive on any host:
- `vrl` — training + Gym reward servers (Ray head/worker, port 6379; Gym Ray port 6380)
- `gym_pass` — standalone pass-rate testing (same Gym Ray port 6380 — cannot run with `vrl`)

---

## 1. Build Images

### Proxy variables

```bash
PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128
NO_PROXY=registry.cn-hangzhou.aliyuncs.com
```

### Build base image (rarely needed, ~30 min)

```bash
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.base \
  --platform linux/amd64 --provenance=false --sbom=false \
  --network host --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base \
  --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY .
```

### Build dev image (for code changes, ~10-15 min)

```bash
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.ncr.26.02.mydev \
  --platform linux/amd64 --provenance=false --sbom=false \
  --network host --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
  --build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY .
```

### Pull and re-tag after push

```bash
dvoff   # REQUIRED — Aliyun auth fails through proxy
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
docker tag  registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
            myverl:ncr2602_vllm012.dev
```

### Distribute to b31 (if registry unavailable)

```bash
docker save myverl:ncr2602_vllm012.dev | gzip > /mnt/public/lichang93/myverl_dev.tar.gz
# On b31:
docker load < /mnt/public/lichang93/myverl_dev.tar.gz
```

**Build flag notes:**
- `--provenance=false --sbom=false` — prevents BuildKit OCI attestation entry that Aliyun CCR rejects
- `--build-arg HTTP_PROXY` — HTTP proxy passed into Dockerfile `RUN` commands (for curl/git); PyPI uses Aliyun mirror directly (no proxy needed)

---

## 2. Baremetal Container Lifecycle

**Golden rule: always recreate containers, never reuse.**

### Start single-node (b32 only)

```bash
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down && docker compose up -d head
docker exec vrl ray status   # expect: 1 node, 8 GPUs
```

### Start 2-node (b32 head + b31 worker)

```bash
# b32:
docker compose down && docker compose up -d head
# b31:
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml down && \
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker

# Verify:
docker exec vrl ray status   # expect: 2 nodes, 16 GPUs
```

### Switch from gym_pass → vrl (or vice versa)

```bash
# If gym_pass was running, before starting vrl:
docker compose -f gym_rollout/docker-compose.yml down
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

---

## 3. Baremetal Training

### Start Gym reward servers (any single node)

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/gym/start_gym_uv.sh
# --status: check only
# --stop:   stop servers + Gym Ray cluster
```

Starts Gym Ray on port 6380 + 7 reward servers on ports 20001-20007 (code_gen, mcqa, instruction_following, structured_outputs, workplace_assistant, math_with_judge, qydomain).

### Single-node smoke test (Moonlight-16B, 1×8 B300)

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_single_domain.sh
```

Uses DAPO reward (no Gym servers needed). `EP=8`, `moe_enable_deepep=True`, `flex` dispatcher.

### 2-node multi-domain training (Moonlight-16B, 2×8 B300)

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_multi_domain.sh
```

- Starts Gym on both nodes via Ray remote tasks (no SSH needed)
- `EP=16`, **`moe_token_dispatcher_type=alltoall`**, `moe_enable_deepep=False`
- Reward: NemoGym 6 domains, URLs `http://localhost:2000x`
- Timing: ~280s/step at `max_response_length=2048`

> **Why alltoall not flex on baremetal?** NCCL pre-initializes IB RC QPs across all EP workers. When DeepEP Buffer() lazily creates DCT QPs at step 1, IB resources are exhausted → DCT failure. K8s works because pod network is fresh and NVSHMEM_IB_GID_INDEX is set correctly.

### 2-node FAPO+curriculum (40Bra reference)

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_40bra_2node_fapo_curriculum.sh
```

---

## 4. K8s RayJob Training (KubeRay)

**kubectl proxy required** on b32 (k8s API server is at `180.184.249.201:8021`):

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl <command>
```

### Pre-run checklist

```bash
# 1. Create + chmod checkpoint dir
mkdir -p /mnt/public/lichang93/st_verl_dockerfile/ckpts/<experiment_name>
chmod o+rwx /mnt/public/lichang93/st_verl_dockerfile/ckpts/
chmod -R o+rwx /mnt/public/lichang93/st_verl_dockerfile/ckpts/<experiment_name>

# 2. Ensure scripts are readable by k8s pods (PFS squashes root → other UID)
chmod o+rx verl/my_scripts/k8s/*.sh
chmod o+rx verl/my_scripts/gym/*.sh

# 3. Verify image tag in RayJob YAML matches pushed image
grep "image:" iter_kuberay_32nodes_verl_training/yaml/<stage>.yaml
```

### Stage 9 — 32-node FAPO+curriculum (COMPLETED)

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
kubectl apply -f iter_kuberay_32nodes_verl_training/yaml/stage9-40bra-32node-fapo-curriculum-rayjob.yaml
```

Entrypoint: `run_40bra_k8s_32node_fapo_curriculum.sh`
- 32 nodes, 256 GPUs, `train_prompt_bsz=256`
- FAPO: `delta=0.10, tau_pos=4.0, tau_neg=5.0`
- Curriculum: Gaussian decay `0.8 → 0.2` over 600 steps
- DeepEP flex dispatcher (works in k8s, `NVSHMEM_IB_GID_INDEX=5` set)
- Completed 214 steps. Checkpoints at steps 100/200/214.

### Stage 10 — 16-node single-domain ablations (CURRENT RUN)

```bash
# Math combo (c351):
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
kubectl apply -f iter_kuberay_32nodes_verl_training/yaml/stage10a-40bra-16node-sd-math-rayjob.yaml

# Code combo (c361):
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
kubectl apply -f iter_kuberay_32nodes_verl_training/yaml/stage10b-40bra-16node-sd-code-rayjob.yaml
```

Entrypoint: `run_40bra_k8s_16node_single_domain.sh c351` (or `c361`)

**Hydra config composition** (Stage 10 migration away from 130-arg bash scripts):
```bash
python3 -m verl.trainer.main_ppo \
  --config-path=/root/myCodeLab/host/verl/my_scripts/k8s/config \
  --config-name=40bra_16node_sd \
  40bra_combo=c351 \
  trainer.experiment_name=bra40-16node-sd-math \
  data.train_files=[...] \
  data.val_files=[...] \
  trainer.default_hdfs_dir=...
```

Static config: `40bra_16node_sd.yaml` (parallelism, batch sizes, curriculum, Gym URLs, optimizer)
Combo config: `40bra_combo/c351.yaml` (FAPO params, domain, ppo_epochs)

**Active Stage 10 params (c351)**:
- FAPO: `delta=0.5, tau_pos=1.5, tau_neg=2.0`
- `max_response_length=16384` (reduced from 30960 — OOM fix)
- `n_resp=16, train_prompt_bsz=128`
- Domain: `nemogym_math` (curriculum sampler `domains=nemogym_math domain_balanced=false`)
- Val file: `0320_split/eval_nemogym_math.parquet` (125 rows, ~8 min val)
- Step timing: ~400-450s/step

### Monitor running jobs

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl describe rayjob <name>
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl logs <pod-name>
```

---

## 5. K8s vs Baremetal — Key Differences

| Config | Baremetal | K8s |
|---|---|---|
| Network interface | `bond0` | `eth0` |
| `NCCL_IB_GID_INDEX` | `3` | `5` |
| `NCCL_IB_TC` | `138` | `96` |
| `NVSHMEM_IB_GID_INDEX` | unset | `5` (critical!) |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `cpu_host_memory` | `gpu` |
| `NCCL_NVLS_ENABLE` | unset | `0` |
| DeepEP flex dispatcher | BLOCKED (IB resource conflict) | WORKS (Stage 9 confirmed) |
| Gym startup | `start_gym_uv.sh` direct | Ray remote `NodeAffinitySchedulingStrategy` |
| PFS permissions | root | squashed to other UID → need `o+rx` on scripts, `o+rwx` on ckpt dirs |

---

## 6. Gym Pass-Rate Standalone Tests

```bash
# Stop vrl first, then:
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym

docker compose -f gym_rollout/docker-compose.yml up -d
docker exec -it gym_pass bash
bash /root/myCodeLab/host/gym_rollout/entrypoint.sh   # starts vLLM + Gym + generates
# or: bash entrypoint.sh --gen-only  (if already started)

# Test all 7 domains:
docker exec gym_pass python3 /root/myCodeLab/host/gym_rollout/test_all_domains.py
```

---

## 7. Common Operations

```bash
# Check Ray cluster status
docker exec vrl ray status

# List Ray jobs
docker exec vrl ray job list

# Follow Ray job logs
docker exec vrl ray job logs <job_id>

# Follow Gym logs
docker exec vrl tail -f /tmp/ng_run_gym.log

# Check Gym server health (all 7 ports)
for p in 20001 20002 20003 20004 20005 20006 20007; do
  curl -s http://localhost:$p/ && echo "port $p OK" || echo "port $p FAIL"
done
```

---

## 8. File Map — Where to Find What

### Build system

| What | File |
|---|---|
| Base image layers (vLLM + CUDA + system deps) | `Dockerfile.base` |
| Dev image layers (verl + Gym venvs) | `Dockerfile.ncr.26.02.mydev` |
| 2-node baremetal orchestration (env vars, mounts, ULimits) | `docker-compose.yml` |
| Standalone pass-rate container | `gym_rollout/docker-compose.yml` |
| Architecture overview + layer rationale | `readme.md` |

### Baremetal training

| What | File |
|---|---|
| 1-node smoke (DAPO reward, DeepEP flex) | `verl/my_scripts/run_single_domain.sh` |
| 2-node multi-domain (alltoall, NemoGym) | `verl/my_scripts/run_multi_domain.sh` |
| 2-node FAPO+curriculum (40Bra reference) | `verl/my_scripts/run_40bra_2node_fapo_curriculum.sh` |
| 2-node single-domain smoke (Stage 10 verification) | `verl/my_scripts/run_40bra_2node_single_domain_smoke.sh` |
| Gym startup/stop/status | `verl/my_scripts/gym/start_gym_uv.sh` |
| Start Gym on all Ray nodes (remote tasks) | `verl/my_scripts/gym/start_gym_all_nodes.py` |
| Ray runtime env (baremetal NCCL vars) | `verl/my_scripts/my_deepep_env.yaml` |

### K8s training

| What | File |
|---|---|
| **Stage 9** entrypoint (32-node FAPO+curriculum) | `verl/my_scripts/k8s/run_40bra_k8s_32node_fapo_curriculum.sh` |
| **Stage 10** entrypoint (16-node single-domain, CURRENT) | `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh` |
| Stage 9 RayJob YAML (32-node, 256 GPUs) | `iter_kuberay_32nodes_verl_training/yaml/stage9-40bra-32node-fapo-curriculum-rayjob.yaml` |
| Stage 10a RayJob YAML (16-node math, c351) | `iter_kuberay_32nodes_verl_training/yaml/stage10a-40bra-16node-sd-math-rayjob.yaml` |
| Stage 10b RayJob YAML (16-node code, c361) | `iter_kuberay_32nodes_verl_training/yaml/stage10b-40bra-16node-sd-code-rayjob.yaml` |
| Ray runtime env (k8s NCCL + NVSHMEM vars) | `verl/my_scripts/k8s/my_deepep_env_k8s.yaml` |

### Hydra configs

| What | File |
|---|---|
| Base PPO trainer config (full schema) | `verl/verl/trainer/config/ppo_megatron_trainer.yaml` |
| 40Bra 16-node static config (Stage 10) | `verl/my_scripts/k8s/config/40bra_16node_sd.yaml` |
| Cluster / hardware / paths (16-node k8s) | `verl/my_scripts/k8s/config/env_k8s_b300_16node.yaml` |
| All combo definitions (c351/c361/c352...) | `verl/my_scripts/k8s/config/combo_40Bra.yaml` |
| Per-combo YAML overrides | `verl/my_scripts/k8s/config/40bra_combo/<id>.yaml` |
| Data / curriculum config schema | `verl/verl/trainer/config/data/legacy_data.yaml` |

### Operations, bugs, and stage history

| What | File |
|---|---|
| **Complete operator guide** (build, run, known issues) | `CLAUDE.md` |
| **Stage-by-stage progression** (Stages 0-10 plan + status) | `iter_kuberay_32nodes_verl_training/training_plan.md` |
| **Full bug log** (root causes, fixes, metrics) | `iter_kuberay_32nodes_verl_training/dev_notes.md` |
| One-time setup + pod spec template | `iter_kuberay_32nodes_verl_training/quickstart.md` |
| Stage 10 plan and current dev notes | `verl/plans/single_domain_trainer/dev_notes.md` |
| Curriculum sampler refactor notes | `verl/plans/curriculum/refactor_sampler_to_1570827b.md` |

### Gym

| What | File |
|---|---|
| Gym server port mapping (ng_run config) | `gym_config/gym_blend_servers.yaml` |
| Per-server venv layout (baked at `/opt/gym_venvs/`) | `Dockerfile.ncr.26.02.mydev` |
| Pass-rate generation + test pipeline | `gym_rollout/entrypoint.sh` |
| Domain correctness test (ports 20001-20007) | `gym_rollout/test_all_domains.py` |

### Critical source patches

| What | File | Line |
|---|---|---|
| Checkpoint sharding fix (`dp_reshardable`) | `verl/verl/utils/checkpoint/megatron_checkpoint_manager.py` | 269 |
| Tokenizer race fix (PFS `sys.modules` corruption) | `verl/verl/models/transformers/hf_tokenizer.py` | (retry loop) |
| DeepEP fused_a2a patch (NVLink=False, NVLink_bytes=0) | Baked into `Dockerfile.base` COPY+patch step | — |

---

## 9. Currently Running Scripts (2026-03-27)

**Stage 10 — 40Bra 16-node single-domain, 4th submission attempt (resubmitted 2026-03-26 15:42 UTC)**

Two concurrent jobs:

| Job | Combo | Domain | RayJob name | YAML |
|---|---|---|---|---|
| Math | c351 | `nemogym_math` | `bra40-16node-sd-math` | `stage10a-40bra-16node-sd-math-rayjob.yaml` |
| Code | c361 | `nemogym_code` | `bra40-16node-sd-math-p1` | `stage10b-40bra-16node-sd-code-rayjob.yaml` |

Both entrypoints call: `run_40bra_k8s_16node_single_domain.sh <combo_id>`

**Check status:**
```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob | grep bra40-16node-sd
```
