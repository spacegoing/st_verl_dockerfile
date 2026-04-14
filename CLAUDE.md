# CLAUDE.md — st_verl_dockerfile Operator Guide

## 1. Project Overview

This repo builds and operates Docker images for **40Bra (40B) and Moonlight-16B GRPO training** using the verl RL training framework. Primary target is the **32-node B300 cluster via KubeRay** (k8s); baremetal 2-node B300 SXM6 is used for smoke tests and development.

> **K8s training docs**: `iter_kuberay_32nodes_verl_training/training_plan.md` — stage-by-stage progression, script layout, and all k8s-specific bugs/fixes.
> **Current active run**: Stage 7 — 40Bra 32-node multi-domain Gym training (256 GPUs, nemogym_server reward).

### Two-Image Architecture

| Image | Dockerfile | Purpose | Rebuild frequency |
|---|---|---|---|
| `myverl:ncr2602_vllm012.base` | `Dockerfile.base` | vLLM build, CUDA extensions, system deps | Rarely (~30 min, CUDA compile) |
| `myverl:ncr2602_vllm012.dev` | `Dockerfile.ncr.26.02.mydev` | verl code, Gym venvs (FROM base) | Code changes (~10-15 min) |

Registry: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl`

### Cluster Layout

| Host | IP | Role | Run from |
|---|---|---|---|
| b32 | 10.12.11.6 | Ray head (verl training head) | Here (Claude Code runs on b32) |
| b31 | 10.12.11.5 | Ray worker | SSH required |

**Mount design (v2 — no symlinks):**
- `/mnt/public/lichang93/st_verl_dockerfile` → `/root/myCodeLab/host` inside container
- `/mnt/public/lichang93/downloads` → `/root/myCodeLab/host/downloads` inside container
- verl is installed as an editable install (`pip install -e`) pointing to the PFS mount — all nodes see the same source code without needing `working_dir` in the Ray runtime env.

### B300 GPU Specifics

- Compute capability: **sm_103** (not sm_100) — vLLM's `is_device_capability(100)` check fails without `VLLM_ATTENTION_BACKEND=CUTLASS_MLA`
- cuDNN 9.18 (MLA fused attention requires >= 9.11)
- RoCE NICs: `mlx5_10` through `mlx5_17`, `NCCL_IB_GID_INDEX=3` (RoCEv2)

---

## 2. Container Lifecycle Rules

### Golden rule: ALWAYS use a fresh container — never reuse

**Every training run, every Gym restart, every time: down+up the container. No exceptions.**

Reusing a live container across runs causes internal state (Ray sessions, stale processes, log files) to accumulate and corrupt. Lesson learned: a `vrl` container left running for 3 days while `start_gym_uv.sh` was called repeatedly inside it accumulated 2.1TB of Ray log spam from orphaned `gcs_server`/`raylet` processes — filling the entire 3.4TB local disk and breaking the PFS mount.

A fresh container costs nothing (seconds to recreate) and eliminates an entire class of hard-to-diagnose bugs.

```bash
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down && docker compose up -d head   # on b32
```

Use `docker compose down` (not `docker rm -f vrl`) — `down` also cleans up the compose-managed network, whereas `rm -f` leaves it behind.

**On b31 too — always recreate both nodes before a 2-node run:**
```bash
# b32:
docker compose down && docker compose up -d head
# b31:
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml down && \
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker
```

### vrl and gym_pass are mutually exclusive

Both containers use `network_mode: host` and both bind **port 6380** (Gym's isolated Ray cluster). They cannot run at the same time on the same node.

**Before starting `vrl` (if `gym_pass` was running):**

```bash
docker compose -f gym_rollout/docker-compose.yml down
# Kill any Ray processes still holding port 6380 on the host (they survive container stops)
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

**Before starting `gym_pass` (if `vrl` was running):**

```bash
docker compose down
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

### Port 6380 session mismatch

Ray persists session state in `/tmp/ray_gym`. If a container is recreated without wiping this directory, Ray's `gcs_server` throws:

```
AssertionError: Session name X does not match persisted value Y
```

Fix: always `rm -rf /tmp/ray_gym` before starting any container that launches the Gym Ray cluster. `start_gym_uv.sh` does this automatically on startup.

### Gym Ray log disk exhaustion ("Wrong cluster ID token" spam)

**Root cause: reusing a container across multiple training runs.** This should not happen if you follow the golden rule (rm+recreate every time). The accumulation only occurs when `start_gym_uv.sh` is called repeatedly inside a long-lived container without a full container restart.

**Symptom:** `/tmp/ray_gym/session_.../logs/raylet.out` and `gcs_server.out` grow to hundreds of GB or more, filling local disk. Each line is:
```
(gcs_server) server_call.h:228: Wrong cluster ID token in request! Expected: X, but got: Y
```

**Mechanism:** `pkill -9 -f /opt/gym_venvs` kills the Gym server actors but NOT `gcs_server`/`raylet` (Ray internals that live in Ray's own venv). Orphaned processes from old sessions spam the new GCS at ~GB/hour with no rate limiting.

**Prevention: just recreate the container.** All processes die on `docker compose down`.

**Defense-in-depth fix (applied in `start_gym_uv.sh`):** Added `pkill -9 -f "${GYM_RAY_TEMP}"` to also kill Ray internals by their temp-dir path — catches `gcs_server`/`raylet` regardless of venv.

**Manual recovery** (if disk already full):
```bash
docker exec vrl pkill -9 -f gcs_server
docker exec vrl pkill -9 -f raylet
docker exec vrl pkill -9 -f /opt/gym_venvs
docker exec vrl pkill -9 -f /tmp/ray_gym
docker exec vrl rm -rf /tmp/ray_gym
```

---

## 3. Building Images

### Proxy variables (set once, reuse)

```bash
PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128
NO_PROXY=registry.cn-hangzhou.aliyuncs.com
```

### Build and push base image (rarely needed)

Only rebuild when vLLM version, CUDA extensions, or system packages change (~30 min):

```bash
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.base \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --network host \
  --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base \
  --build-arg HTTP_PROXY=$PROXY \
  --build-arg HTTPS_PROXY=$PROXY \
  .
```

### Build and push dev image (for verl/Gym code changes)

Rebuild whenever verl or Gym code changes (~10-15 min, no CUDA compile):

```bash
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.ncr.26.02.mydev \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --network host \
  --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
  --build-arg HTTP_PROXY=$PROXY \
  --build-arg HTTPS_PROXY=$PROXY \
  .
```

After pushing, re-tag locally for docker-compose (which uses the short name):

```bash
# dvoff first — Aliyun registry auth fails through proxy
dvoff
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
docker tag  registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
            myverl:ncr2602_vllm012.dev
```

### Why these build flags

- `--provenance=false --sbom=false`: prevents BuildKit from wrapping the image in an OCI manifest list with an attestation entry. Aliyun CCR rejects or mishandles the extra `unknown/unknown` entry.
- `--platform linux/amd64`: ensures a single-platform manifest.
- `--build-arg HTTP_PROXY`: applies proxy inside `RUN` commands (apt-get, git clone, curl for uv install). PyPI downloads use `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` directly — no proxy needed for those.

### Distribute image to b31 (if registry pull is unavailable)

```bash
docker save myverl:ncr2602_vllm012.dev | gzip > /mnt/public/lichang93/myverl_dev.tar.gz
# On b31:
docker load < /mnt/public/lichang93/myverl_dev.tar.gz
```

---

## 4. Starting the Training Container (vrl)

The `vrl` container is the Ray head node for verl training. It starts Ray and keeps the container alive.

### On b32 (head node — where we run Claude Code)

```bash
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down && docker compose up -d head
```

This starts Ray with `--head --port=6379 --node-ip-address=10.12.11.6`.

### On b31 (worker node, 2-node setup only)

```bash
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml down && \
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker
```

This starts Ray pointing at head: `--address=10.12.11.5:6379`.

### Verify Ray cluster status

```bash
docker exec vrl ray status
```

Expected: 2 nodes, 16 GPUs total (8 per node) for 2-node setup; 1 node, 8 GPUs for single-node.

---

## 5. Starting Gym Resource Servers

Gym servers run inside whichever container is active (`vrl` for training, `gym_pass` for standalone pass rate tests). The script manages its own isolated Ray cluster on port 6380.

### Script location

Inside container: `/root/myCodeLab/host/verl/my_scripts/start_gym_uv.sh`

### Commands

```bash
# Start Ray (port 6380) + all 7 Gym servers (ports 20001-20007), wait for health
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/start_gym_uv.sh

# Check status only
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/start_gym_uv.sh --status

# Stop Gym's Ray cluster + servers
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/start_gym_uv.sh --stop
```

### What it does

1. Kills any stale processes on port 6380 and wipes `/tmp/ray_gym` (prevents session mismatch).
2. Starts an isolated Ray head on port 6380 with ports fixed in the 7300-7999 range (outside verl's 10002-19999 worker range, no dashboard to save ~1.1 GB RAM).
3. Patches `/opt/gym_config/gym_blend_servers.yaml` to add `default_host: "0.0.0.0"` (idempotent). The file is baked into the image with no `default_host` key, so uvicorn defaults to binding `127.0.0.1` (loopback only). The patch makes servers reachable from external machines (e.g. cross-node monitoring: `curl http://10.12.11.5:20001/` from b32). Reward workers use `localhost` URLs and work with either binding — this is for observability, not correctness.
4. Launches all 7 Gym resource servers via `ng_run` using `/opt/gym_config/gym_blend_servers.yaml`.
5. Polls all 7 ports (20001-20007) until healthy (timeout: 120s).

### Log location

```bash
tail -f /tmp/ng_run_gym.log
```

### Venv layout (baked into image)

- `/opt/gym_venvs/main/` — main venv: `ng_run`, `ray`, `nemo_gym`
- `/opt/gym_venvs/resources_servers/<name>/.venv/` — per-server isolated venvs
- No `.venv` dirs in the Gym repo on PFS (avoids PFS write pressure)

---

## 6. Running verl Training

### Single-node smoke test (Moonlight-16B, 8x B300)

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_moonlight_1node_blend_smoke.sh
```

**What it does:**
1. Calls `start_gym_uv.sh` to bring up all 7 Gym reward servers.
2. Sets up wandb (offline mode, stored in `/root/myCodeLab/host/verl/wandb_my_dirs/`).
3. Submits via `RAY_ADDRESS='auto' ray job submit --runtime-env=my_deepep_env.yaml`.
4. Runs `verl.trainer.main_ppo` with Hydra config `ppo_megatron_trainer.yaml`.

**Key training parameters (smoke test defaults):**
- `train_prompt_bsz=8`, `n_resp_per_prompt=2`, `ppo_epochs=1`
- `NNODES=1`, `gen_tp=1`, `train_tp=1`, `train_pp=1`, `EP=8`
- `max_prompt_length=4096`, `max_response_length=2048`
- Reward manager: `nemogym_server` hitting ports 20001-20006

**Log location:**

```bash
ls logs/blend_smoke_*.log      # inside container, from /root/myCodeLab/host/verl/
tail -f logs/blend_smoke_<TIMESTAMP>.log
```

---

### 2-Node training (Moonlight-16B, 2x8 B300, EP=16)

**Prerequisites:**
- `vrl` container running on b32: `docker compose up -d head`
- `vrl` container running on b31: `docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker`
- `docker exec vrl ray status` shows 2 nodes / 16 GPUs

**Launch (from b32, outside container):**

```bash
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_multi_domain.sh
```

The script starts Gym on the head node locally, then dispatches `start_gym_uv.sh` to all worker nodes via Ray remote tasks (`start_gym_all_nodes.py`) — same pattern as k8s. No SSH/pdsh dependency.

**Key parameters vs 1-node:**
- `train_prompt_bsz=16` (doubled), `EP=16` (all 16 GPUs), `NNODES=2`
- Reward URLs: `http://localhost:2000x` — each node runs its own Gym, reward workers hit localhost
- MoE dispatcher: `alltoall` + `moe_enable_deepep=False` (DeepEP/NVSHMEM not used cross-node; see Known Issues)
- `NVSHMEM_BOOTSTRAP=IB` still in `my_deepep_env.yaml` — kept for potential future DeepEP use

**2-node NCCL config (all 8 NICs, mlx5_14 fixed by admin):**
- `NCCL_IB_HCA: mlx5_10-17` (all 8 NICs, GID[3] routable on both nodes)
- Set in both `docker-compose.yml` (container-level) and `my_deepep_env.yaml` (Ray worker-level)

**Dual Ray cluster isolation (per node):**
- Port 6379 — verl Ray (cross-node, 2 nodes, 16 GPUs total)
- Port 6380 — Gym Ray (local-only, 1 node per host, isolated via `--temp-dir=/tmp/ray_gym`)
- Each node has its own Gym Ray; reward workers hit `localhost:2000x` — never cross-node for reward

**Verify dual cluster status:**
```bash
docker exec vrl ray status                                                    # verl Ray: 2 nodes
docker exec vrl /opt/gym_venvs/main/bin/ray status --address=127.0.0.1:6380  # b32 gym: 1 node
ssh b31 "docker exec vrl /opt/gym_venvs/main/bin/ray status --address=127.0.0.1:6380"  # b31 gym: 1 node
```

**Check Ray job progress:**

```bash
docker exec vrl ray job list   # get job ID
docker exec vrl ray job logs <job_id>
```

**Per-step timing:** ~280s/step at `train_prompt_bsz=16`, `max_response_length=2048`.

---

### Ray job submit flow

```bash
RAY_ADDRESS='auto' ray job submit --runtime-env="${RUNTIME_ENV}" -- python3 -m verl.trainer.main_ppo ...
```

`RAY_ADDRESS='auto'` connects to the Ray cluster already running in the container. The runtime env is `verl/my_scripts/my_deepep_env.yaml`.

**Note:** `working_dir` is intentionally absent from `my_deepep_env.yaml`. verl is installed as an editable install via PFS (`pip install -e /root/myCodeLab/host/verl`), so all Ray worker nodes pick up the code through the shared filesystem automatically. No working_dir packaging needed.

### Check Ray job logs

```bash
docker exec vrl ray job logs <job_id>
docker exec vrl ray job list   # list all jobs + status
```

---

## 7. Running Gym Rollout Tests (Standalone Pass Rate)

These tests use the `gym_pass` container, which is mutually exclusive with `vrl`.

### Start the gym_pass container

```bash
# Must not have vrl running — stop it first
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym

docker compose -f gym_rollout/docker-compose.yml up -d
docker exec -it gym_pass bash
```

### Full pass rate generation (vLLM + Gym + generate)

Inside the `gym_pass` container:

```bash
bash /root/myCodeLab/host/gym_rollout/entrypoint.sh
```

**What it does:**
1. Starts vLLM serving Moonlight-16B with DP=8, TP=1 on port 8000 (waits up to 600s for ready).
2. Calls `start_gym_uv.sh` to start all 7 Gym servers on ports 20001-20007.
3. Runs `generate_pass_rates.py` with `--n-samples 10 --concurrency 256`.
4. Outputs to `/root/myCodeLab/host/gym_rollout/output/`.

**Skip vLLM/Gym start if already running:**

```bash
bash /root/myCodeLab/host/gym_rollout/entrypoint.sh --gen-only
```

**Stop everything:**

```bash
bash /root/myCodeLab/host/gym_rollout/entrypoint.sh --stop
```

**Logs:**

```bash
tail -f /root/myCodeLab/host/gym_rollout/logs/vllm.log
tail -f /root/myCodeLab/host/gym_rollout/logs/generate_*.log
```

### Test all 7 Gym domain servers

Run after servers are healthy to verify each domain returns correct rewards:

```bash
docker exec gym_pass python3 /root/myCodeLab/host/gym_rollout/test_all_domains.py
```

Tests ports 20001-20007 (code_gen, mcqa, instruction_following, structured_outputs, workplace_assistant, math_with_judge, qydomain) with correct and wrong answer pairs. All should return `[OK]`.

---

## 8. Proxy Usage

### Shell proxy (von/voff)

Wraps pip, curl, git, etc. in the current shell session:

```bash
von    # enable proxy
voff   # disable proxy
```

Use `von` for: downloading packages, cloning repos, curl to external URLs.

### Docker daemon proxy (dvon/dvoff)

Controls proxy for the Docker daemon itself. **Restarts Docker** — use sparingly:

```bash
dvon   # enable daemon proxy (for docker pull from nvcr.io etc.)
dvoff  # disable daemon proxy (REQUIRED before docker push to Aliyun)
```

Use `dvon` for: `docker pull` from external registries (nvcr.io, docker.io).

**CRITICAL:** Use `dvoff` before `docker push` to Aliyun CCR (`registry.cn-hangzhou.aliyuncs.com`). Aliyun registry auth fails through the proxy.

**Do NOT use** `dvon`/`dvoff` for `docker build` — proxy for build `RUN` commands is passed via `--build-arg HTTP_PROXY=...` instead.

### Proxy throughput

~33-96 KB/s — too slow for PyPI downloads during docker build. Use `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` (already set in both Dockerfiles) for fast mainland China PyPI access. Proxy is still needed for `curl astral.sh` (uv install) and `git clone` operations.

---

## 9. Key Environment Variables

These are set in `docker-compose.yml` (for `vrl`) and `gym_rollout/docker-compose.yml` (for `gym_pass`):

| Variable | Value | Why |
|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `CUTLASS_MLA` | B300 is sm_103; vLLM's sm_100 check fails without this |
| `VLLM_USE_V1` | `1` | Use vLLM v1 engine |
| `PYTHONPYCACHEPREFIX` | `/tmp/pycache` | Redirect pyc writes to local disk; verl and nemo_gym are editable PFS installs so __pycache__ dirs would otherwise accumulate on the parallel filesystem |
| `NCCL_IB_GID_INDEX` | `3` | RoCEv2 on B300 cluster (mlx5_10-17) |
| `NCCL_IB_HCA` | `mlx5_10:1,...,mlx5_17:1` | GPU-affine RoCE NICs only |
| `NVSHMEM_IB_ENABLE_IBGDA` | `1` | DeepEP IBGDA NIC handler |
| `NVTE_FUSED_ATTN` | `1` | Transformer Engine fused attention (MLA) |
| `WANDB_MODE` | `offline` | No outbound wandb.ai connection |
| `RAY_USAGE_STATS_ENABLED` | `0` | Disable Ray telemetry |

Variables set in `my_deepep_env.yaml` (Ray runtime env, propagated to all Ray workers):

| Variable | Value | Why |
|---|---|---|
| `RAY_OVERRIDE_JOB_RUNTIME_ENV` | `1` | Allow job-level env to override cluster-level |
| `VLLM_USE_V1` | `1` | Same as above, propagated to worker processes |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Required for Megatron tensor parallelism |
| `HYDRA_FULL_ERROR` | `1` | Full Hydra config error output |
| `OTEL_SDK_DISABLED` | `true` | Disable OpenTelemetry SDK |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `cpu_host_memory` | DeepEP: IBRC QPs (not DCT) — this cluster's IB fabric doesn't support DCT |

**Note:** `working_dir` is NOT set in `my_deepep_env.yaml`. verl is installed as an editable package pointing to the PFS mount (`/root/myCodeLab/host/verl`), which is accessible from all cluster nodes via the shared filesystem. Packaging a working_dir tarball would be redundant and slow.

---

## 10. Known Issues and Fixes

### Port 6380 Ray session mismatch

**Symptom:** Ray `gcs_server` crashes with:
```
AssertionError: Session name X does not match persisted value Y
```

**Cause:** Gym Ray processes survive container restarts (host network mode). `fuser -k -9 6380/tcp` only kills the port holder; other gym Ray processes (GCS clients, monitor, worker agents) keep GCS session data in memory. When a new `ray start --head` tries to write a new session to the GCS, it finds the old session already stored → mismatch.

**Fix:** `start_gym_uv.sh` now kills ALL gym venv processes, not just the port holder:
```bash
pkill -9 -f /opt/gym_venvs 2>/dev/null || true
fuser -k -9 6380/tcp 2>/dev/null || true
sleep 2
rm -rf /tmp/ray_gym
```

For manual recovery (if session mismatch occurs despite the above):
```bash
# Preferred: full container cycle
cd /mnt/public/lichang93/st_verl_dockerfile
docker compose down
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
docker compose up -d head
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/gym/start_gym_uv.sh
```

### PFS pyc cache accumulation

**Symptom:** `/root/myCodeLab/host/verl/__pycache__` and `Gym/__pycache__` dirs grow on the parallel filesystem, causing slow directory listings and potential NFS/PFS metadata pressure.

**Fix:** `PYTHONPYCACHEPREFIX=/tmp/pycache` redirects all `.pyc` file writes to local `/tmp`. Already set in both docker-compose files.

**Side effect in gym_pass:** On fresh container start, Python must recompile venv source to `/tmp/pycache` — this is fast since venv source is on local disk.

### working_dir removed from Ray runtime env

**History:** `my_deepep_env.yaml` previously had `working_dir: /root/myCodeLab/host/verl` to distribute verl source to Ray workers. This is now unnecessary because verl is installed as an editable package (`pip install -e .`) into the system venv, and all nodes share the same PFS mount at `/root/myCodeLab/host/verl`.

**Current state:** `working_dir` is absent from `my_deepep_env.yaml`. Do not add it back unless moving away from PFS editable installs.

### vrl and gym_pass port conflict

**Symptom:** Second container fails to start or Ray fails to bind port 6380.

**Cause:** Both containers use `network_mode: host` and both `start_gym_uv.sh` / `entrypoint.sh` start a Ray head on port 6380.

**Fix:** Only one of `vrl` or `gym_pass` can run at a time. Stop and remove the other before starting. See Section 2 for exact commands.

### 2-node: Gym servers not started on b31

**Symptom:** `Cannot connect to host localhost:20002 ... [Errno 111] Connect call failed`

**Cause:** `RewardLoopWorker` actors land on all verl Ray nodes (round-robin). Each hits `localhost:2000x` on its own node. If b31 Gym is not running, b31 workers fail.

**Fix:** `run_multi_domain.sh` now starts Gym on all nodes via Ray remote tasks (`start_gym_all_nodes.py`) — same pattern as k8s. No SSH/pdsh dependency. Head starts locally, workers via Ray `NodeAffinitySchedulingStrategy`.

### 2-node: NVSHMEM IBGDA DCT failure (DeepEP flex cross-node) — FULLY FIXED

There are two layered issues, both fixed permanently in `Dockerfile.base`:

#### Layer 1: NVSHMEM IBGDA NIC handler

**Symptom:** Step 1 crashes with `create DCT share err` / `Unable to create ah`.

**Cause:** Default `NVSHMEM_IBGDA_NIC_HANDLER=gpu` uses DCT QPs. This cluster doesn't support DCT.

**Fix:** `NVSHMEM_IBGDA_NIC_HANDLER: cpu_host_memory` in both `docker-compose.yml` and `my_deepep_env.yaml`. Already set — do not change.

#### Layer 2: NCCL/NVSHMEM IB resource conflict in Megatron (UNRESOLVED — using alltoall instead)

**Symptom:** Step 1 crashes with `ibgda.cpp:2966: non-zero status: 7 create DCT share err` even with `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory`. Happens at first MoE dispatch (`compute_log_prob`), not during initialization.

**Root cause:** NCCL initializes IB RC QPs for all EP workers (16 processes × 8 NICs × 8 QPs/connection = many QPs). When DeepEP `Buffer()` is created lazily at step 1, it calls `sync()` → `nvshmemx_init_attr(IBGDA)`. IBGDA transport initialization requires DCT QP creation on the IB fabric, but IB resources (DCT QP table or context) are exhausted/blocked by NCCL's existing connections → status 7 (EPERM/resource limit).

The standalone DeepEP test PASSES because it starts fresh torchrun processes with NO NCCL initialized — there are no competing IB resources.

Additionally, `buffer.py` forcefully sets `os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'` before NVSHMEM init, so there's no way to disable IBGDA without patching `buffer.py`.

**Partial fix applied** (in Dockerfile.base and both containers):
- `fused_a2a.py`: `allow_nvlink_for_normal_mode=False` → unique NVSHMEM ranks per process
- `fused_a2a.py`: `num_nvl_bytes=0` → no NVLink buffer allocation
- These prevent bootstrap_uid truncation errors but do NOT prevent the DCT failure (different failure mode)

**Working config (CONFIRMED 2026-03-23):** Use `alltoall` dispatcher (`moe_token_dispatcher_type=alltoall`, `moe_enable_deepep=False`). NVSHMEM is not used, NCCL handles MoE alltoall. Step 1 verified: `actor/grad_norm:8.39`, `timing_s/step:342s` (~5.7 min/step), no DCT errors. Performance penalty vs DeepEP flex is acceptable for smoke tests.

**Future DeepEP flex enablement:** Pre-initialize DeepEP Buffer BEFORE NCCL initialization in the Megatron worker startup (not lazily). This would avoid the IB resource conflict. Requires Megatron-Bridge code changes.

**Verification:** Standalone `bash verl/my_scripts/run_deepep_test.sh cpu` passes — NVSHMEM/IBGDA works when NCCL is not competing.

### 2-node: NVSHMEM bootstrap failure (DeepEP flex cross-node)

**Symptom:** Workers hang silently after `WorkerDict` actor init (no output, GPUs idle).

**Cause:** Default NVSHMEM bootstrap (`UID` socket) requires a shared `/tmp` between nodes — fails cross-node. IB bootstrap is needed.

**Fix:** `NVSHMEM_BOOTSTRAP: "IB"` in `my_deepep_env.yaml`. Already set — do not remove.

### Aliyun push fails through proxy

**Symptom:** `docker push registry.cn-hangzhou.aliyuncs.com/...` fails with auth error.

**Fix:** Run `dvoff` before pushing. Aliyun CCR auth cannot traverse the proxy.

### BuildKit OCI manifest list rejection by Aliyun

**Symptom:** `docker push` succeeds but the image is wrapped in a manifest list with an `unknown/unknown` attestation — Aliyun CCR rejects it or `docker pull` gets a wrong image.

**Fix:** Always use `--provenance=false --sbom=false` in `docker buildx build` commands.
