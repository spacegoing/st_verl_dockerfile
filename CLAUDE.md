# CLAUDE.md — st_verl_dockerfile Operator Guide

## 1. Project Overview

This repo builds and operates Docker images for **Moonlight-16B GRPO training** using the verl RL training framework on a 2-node B300 SXM6 cluster.

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

### Golden rule: always rm+recreate

Never reuse a stale container. Always stop, remove, then recreate:

```bash
docker rm -f vrl
docker compose up -d head   # on b32
```

### vrl and gym_pass are mutually exclusive

Both containers use `network_mode: host` and both bind **port 6380** (Gym's isolated Ray cluster). They cannot run at the same time on the same node.

**Before starting `vrl` (if `gym_pass` was running):**

```bash
docker stop gym_pass
docker rm -f gym_pass
# Kill any Ray processes still holding port 6380 on the host (they survive container stops)
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

**Before starting `gym_pass` (if `vrl` was running):**

```bash
docker rm -f vrl
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

### Port 6380 session mismatch

Ray persists session state in `/tmp/ray_gym`. If a container is recreated without wiping this directory, Ray's `gcs_server` throws:

```
AssertionError: Session name X does not match persisted value Y
```

Fix: always `rm -rf /tmp/ray_gym` before starting any container that launches the Gym Ray cluster. `start_gym_uv.sh` does this automatically on startup.

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
docker rm -f vrl 2>/dev/null || true
docker compose up -d head
```

This starts Ray with `--head --port=6379 --node-ip-address=10.12.11.6`.

### On b31 (worker node, 2-node setup only)

```bash
docker rm -f vrl 2>/dev/null || true
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
3. Patches `/opt/gym_config/gym_blend_servers.yaml` to add `default_host: "0.0.0.0"` (idempotent). This makes uvicorn bind on all interfaces so cross-node Ray workers on b31 can reach the reward servers on b32. The file is baked into the image with no `default_host` key, which would otherwise cause servers to bind `127.0.0.1` and be unreachable from b31.
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
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_moonlight_2node_blend_smoke.sh
```

The script calls `start_gym_uv.sh` automatically, which now also patches the Gym config
for cross-node access (see Section 5 "What it does", item 3). No manual pre-steps needed.

**Key parameters vs 1-node:**
- `train_prompt_bsz=16` (doubled), `EP=16` (all 16 GPUs), `NNODES=2`
- Reward URLs: `http://10.12.11.6:2000x` (b32 IP — not localhost, workers run on b31 too)
- MoE dispatcher: `alltoall` + `moe_enable_deepep=False` (NVSHMEM/DeepEP flex not working cross-node)

**2-node NCCL constraints:**
- `NCCL_IB_HCA` excludes `mlx5_14` — b31's GID[3] on mlx5_14 is link-local (`fe80::`), not routable. Using mlx5_10-13, mlx5_15-17 (7 NICs, all with routable GID[3] on both nodes).
- Set in both `docker-compose.yml` (container-level) and `my_deepep_env.yaml` (Ray worker-level).

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
docker rm -f vrl 2>/dev/null || true
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym

cd /mnt/public/lichang93/st_verl_dockerfile
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
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | DeepEP: use GPU-side NIC handler |

**Note:** `working_dir` is NOT set in `my_deepep_env.yaml`. verl is installed as an editable package pointing to the PFS mount (`/root/myCodeLab/host/verl`), which is accessible from all cluster nodes via the shared filesystem. Packaging a working_dir tarball would be redundant and slow.

---

## 10. Known Issues and Fixes

### Port 6380 Ray session mismatch

**Symptom:** Ray `gcs_server` crashes with:
```
AssertionError: Session name X does not match persisted value Y
```

**Cause:** `/tmp/ray_gym` contains stale session state from a previous container. With `network_mode: host`, Ray processes can survive container restarts and leave state on the host.

**Fix:** Always wipe before starting:
```bash
fuser -k -9 6380/tcp 2>/dev/null || true
rm -rf /tmp/ray_gym
```

`start_gym_uv.sh` does this automatically at startup.

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

### 2-node: NCCL `ibv_modify_qp` errno 22 on mlx5_14

**Symptom:**
```
ibv_modify_qp failed with 22 Invalid argument on dev mlx5_14:1, local GID index 3,
local GID fe80::8c3d:8ff:fe64:7719
```

**Cause:** b31's mlx5_14 GID[3] is `fe80::` (link-local), not a routable RoCEv2 address. All other NICs have routable GID[3] on both nodes.

**Fix:** `NCCL_IB_HCA` excludes `mlx5_14` in both `docker-compose.yml` and `my_deepep_env.yaml`:
```
mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_15:1,mlx5_16:1,mlx5_17:1
```

### 2-node: RewardLoopWorker on b31 can't connect to reward servers

**Symptom:** `Cannot connect to host localhost:20002 ... [Errno 111] Connect call failed`

**Cause:** Ray's `RewardLoopWorker` actors land on both b31 and b32. b31 has no Gym servers, so `localhost:2000x` fails on b31.

**Fix:** Reward server URLs in `run_moonlight_2node_blend_smoke.sh` use `http://10.12.11.6:2000x` (b32 explicit IP).

### 2-node: Gym servers bound to 127.0.0.1 — unreachable from b31

**Symptom:** Even with correct b32 IP in URLs: `Connect call failed ('10.12.11.6', 20001)`

**Cause:** `/opt/gym_config/gym_blend_servers.yaml` in the image has no `default_host` key. `nemo_gym/global_config.py` defaults to `127.0.0.1`. Note: passing `+default_host=0.0.0.0` as a CLI arg to `ng_run` does NOT work — the key must be in the YAML.

**Fix:** `start_gym_uv.sh` now auto-patches the YAML before launching servers (idempotent):
```bash
sed -i '1a default_host: "0.0.0.0"' /opt/gym_config/gym_blend_servers.yaml
```
No manual action needed.

### 2-node: DeepEP/NVSHMEM flex dispatcher fails cross-node

**Symptom:**
```
allgather of ipc handles failed
nvshmem initialization failed, exiting
```

**Cause:** `moe_token_dispatcher_type=flex` requires NVSHMEM IB bootstrap for cross-node expert routing. NVSHMEM UID socket bootstrap fails on this cluster.

**Fix:** Use `alltoall` dispatcher instead:
```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
```
Already set in `run_moonlight_2node_blend_smoke.sh`. Do not switch to `flex` without first diagnosing NVSHMEM bootstrap.

### Aliyun push fails through proxy

**Symptom:** `docker push registry.cn-hangzhou.aliyuncs.com/...` fails with auth error.

**Fix:** Run `dvoff` before pushing. Aliyun CCR auth cannot traverse the proxy.

### BuildKit OCI manifest list rejection by Aliyun

**Symptom:** `docker push` succeeds but the image is wrapped in a manifest list with an `unknown/unknown` attestation — Aliyun CCR rejects it or `docker pull` gets a wrong image.

**Fix:** Always use `--provenance=false --sbom=false` in `docker buildx build` commands.
