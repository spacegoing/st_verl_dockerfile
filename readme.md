# README — verl Moonlight-16B Training Docker Image (nemo:26.02)

Image: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`
Cluster: b31 (10.12.11.5) + b32 (10.12.11.6), 8x B300 SXM6 each

---

## Directory Structure

```
st_verl_dockerfile/
  Dockerfile.ncr.26.02.mydev   # Active Dockerfile (15 layers on top of nemo:26.02)
  docker-compose.yml            # 2-node cluster orchestration (head + worker)
  readme.md                     # This file
  dev_manual_iter2_gym_server.md  # Iter2 Gym server integration plan, bug log, changelog
  image_deps_report_v2.md       # Exhaustive 3-image dependency diff
  .gitignore / .tmux.conf / to_append.sh / o200k_base.tiktoken  # Static configs
  docs/
    reward_callstack_analysis.md  # verl reward manager callstack analysis
    pip_vs_uv_investigation.md   # Why 26.02 has dual pip/uv, package shadowing analysis
  verl/                         # verl source (editable install)
    my_scripts/                 # Training launch scripts, Gym server launcher
  Gym/                          # NemoGym source (editable install, iter2)
  vllm/                         # vLLM 0.12.0 source (built into image, not editable at runtime)
  mbridge/                      # Megatron-Bridge pip package
  verifiable-instructions/      # Gym dep for instruction_following domain
  nltk_data/                    # NLTK tokenization data
  downloads/                    # Models + datasets (mount: /mnt/public/lichang93/downloads)
    models/Moonlight16B/        # Moonlight-16B weights
    datasets/nemogym_blend/     # Blend dataset (train_v2.parquet, val.parquet)
  legacy/                       # Old Dockerfiles, iter1 docs, old scripts, MJ_NEMO_GYM/, verl.old/
```

---

## Quick Reference

### Build

```bash
# On build machine (b32). Proxy ON for pip downloads.
dvon
DOCKER_BUILDKIT=0 docker build -f Dockerfile.ncr.26.02.mydev --network host \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev .

# Proxy OFF for Aliyun push.
dvoff
docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
```

**Build time**: ~30min cold (vLLM CUDA build), ~2min warm (code-only changes in L10-L13).

**`DOCKER_BUILDKIT=0` required**: The base image has ~107 layers; the legacy builder's
127-layer limit demands consolidation (our 15 layers fit). BuildKit is not used because
its layer counting differs and we hit provenance flag errors.

### Pull (on other node)

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
```

### Deploy cluster

```bash
# b31 (head node):
docker compose up -d head

# b32 (worker node):
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker
```

### Launch training

```bash
docker exec -it vrl bash
cd /root/myCodeLab/host/verl/my_scripts

# 1-node smoke test (8 GPUs, all 6 Gym domains):
bash run_moonlight_1node_blend_smoke.sh

# 2-node full run (16 GPUs):
bash run_moonlight_2node_test.sh
```

### Rebuild after code changes

Editing `verl/` or `Gym/` source **does NOT require rebuild** — they are editable installs
and the docker-compose mount overlays the image's copy with live host source.

Rebuild is only needed when:
- Adding new packages to L5 (use `uv pip install --no-cache --no-deps`)
- Changing `mbridge/`, `vllm/`, or `verifiable-instructions/` source
- Modifying static configs (`o200k_base.tiktoken`, `.tmux.conf`, `to_append.sh`, `nltk_data/`)

**Tip**: To avoid invalidating cache for heavy layers (L1-L4), add small new pip packages
to a new late layer rather than appending to L5.

---

## Dockerfile Layer-by-Layer Rationale

**File**: `Dockerfile.ncr.26.02.mydev`
**Base**: `nvcr.io/nvidia/nemo:26.02` (~107 layers)

Layer order: slowest/most stable first (cache friendly), fastest/most changed last.

### Layer 1 — vLLM prep

```dockerfile
RUN pip uninstall -y vllm 2>/dev/null; \
    uv pip uninstall vllm 2>/dev/null; \
    uv pip install --no-cache setuptools_scm; true
```

26.02 ships vLLM 0.14.2. verl requires exactly 0.12.0 because 0.13+ changed the rollout
worker API. `setuptools_scm` is needed because vLLM's build system requires it, and the
`.git/` directory is stripped from the COPY'd source (to save 1.6GB build context).

Both `pip` and `uv pip` uninstall are used because vllm may exist in both the system
site-packages (`/usr/local/`) and the venv (`/opt/venv/`).

### Layer 2 — COPY vLLM source

```dockerfile
COPY vllm /opt/vllm
```

Source in `vllm/` must be pre-cleaned: delete `.git/`, `.deps/`, `tests/`, `docs/`,
`benchmarks/`, `examples/`, and all `*.so` files. Reduces COPY context from 3.8GB to ~26MB.

### Layer 3 — vLLM CUDA build (~30 min)

```dockerfile
RUN cd /opt/vllm && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0 MAX_JOBS=64 NVCC_THREADS=1 \
    uv pip install --no-deps --no-build-isolation --no-cache -e .
```

Slowest step — placed early so code changes don't trigger recompile.

`--no-build-isolation`: Prevents creating an isolated venv that downloads its own
torch (CUDA 12.6), which would conflict with the base image's torch 2.10.0a0 (CUDA 13.0).

`--no-deps`: Protects all base image packages.

`MAX_JOBS=64 NVCC_THREADS=1`: Maximizes parallel C++ compilation; single-threaded NVCC
(memory-hungry, parallel threads cause OOM).

`-e .` (editable): `/opt/vllm` is NOT under the docker-compose mount, so vLLM always
resolves from the image layer (immutable). This is intentional.

### Layer 4 — CUDA extension packages

```dockerfile
RUN uv pip install --no-cache --no-deps --no-build-isolation grouped_gemm && \
    uv pip install --no-cache --no-deps --no-build-isolation causal_conv1d && \
    uv pip install --no-cache --no-deps --no-build-isolation mamba_ssm
```

Were in nemo:25.11.01 but removed in 26.02. Imported by megatron-core (MoE expert GEMM,
SSM architecture). Each requires NVCC compilation.

### Layer 5 — System packages + pure-Python deps

```dockerfile
RUN apt-get update && apt-get install -y pdsh tmux htop vim && \
    rm -rf /var/lib/apt/lists/* && \
    git config --global --add safe.directory '*' && \
    uv pip install --no-cache --no-deps \
        wandb gpustat codetiming tensordict mathruler pylatexenc torchdata \
        hydra-core bitsandbytes orjson \
        transformers==4.57.3 \
        model_hosting_container_standards anthropic \
        pyvers math_verify latex2sympy2_extended openapi_schema_validator \
        langdetect absl-py immutabledict \
        yappi itsdangerous gprof2dot pydot && \
    wandb login df3cecbfc0874c8a352c40820becf4a15575614e
```

All pure-Python or pre-built wheels — no NVCC needed.

`uv pip install` writes to `/opt/venv/lib/python3.12/site-packages/` (HIGH priority).
This is critical — plain `pip install` would write to `/usr/local/` (LOW priority),
where packages are shadowed by the venv copy. See `docs/pip_vs_uv_investigation.md`.

`--no-deps` for every install: Without it, `wandb` would pull `numpy>=2.0` which
would overwrite the base's `numpy==1.26.4`, breaking every CUDA extension's C ABI.

`transformers==4.57.3`: Overrides the venv's 4.57.6. Now effective because `uv pip`
installs to the HIGH priority location.

`yappi itsdangerous gprof2dot pydot`: Required by Gym's `profiling.py` module-level imports.

`git config --global --add safe.directory '*'`: Prevents "dubious ownership" errors in
mounted volumes.

### Layer 6 — mbridge

```dockerfile
COPY mbridge /tmp/mbridge
RUN cd /tmp/mbridge && uv pip install --no-cache --no-deps . && rm -rf /tmp/mbridge
```

Pure-Python bridge between verl and megatron-core. Non-editable (rarely changes).

### Layer 7 — Static config files

```dockerfile
COPY o200k_base.tiktoken to_append.sh .tmux.conf /tmp/docker_context/
COPY nltk_data /usr/local/share/nltk_data
```

`o200k_base.tiktoken`: vLLM 0.12 uses OpenAI's tiktoken encoding (no internet at runtime).
`nltk_data/`: NLTK tokenization data used by Gym's math evaluation.
`to_append.sh`: Shell aliases. `.tmux.conf`: tmux config.

### Layer 8 — Environment variables

```dockerfile
ENV NLTK_DATA=/usr/local/share/nltk_data \
    TIKTOKEN_ENCODINGS_BASE=/root/tiktoken_cache
```

### Layer 9 — Workspace setup

```dockerfile
RUN mkdir -p /root/myCodeLab/host /root/myCodeLab/public /root/tiktoken_cache && \
    mv /tmp/docker_context/o200k_base.tiktoken /root/tiktoken_cache/ && \
    echo "" >> /root/.bashrc && \
    cat /tmp/docker_context/to_append.sh >> /root/.bashrc && \
    mv /tmp/docker_context/.tmux.conf /root/ && \
    rm -rf /tmp/docker_context && \
    ln -s /opt/venv/lib/python3.12/site-packages /root/myCodeLab/site-packages
```

`mkdir -p /root/myCodeLab/host`: Creates the directory that docker-compose will mount over.
No symlinks, no host path assumptions.

Symlink points to `/opt/venv/` site-packages (the actual runtime location where `uv pip`
installs packages), not `/usr/local/` dist-packages.

### Layers 10-12 — COPY codebases (change most often)

```dockerfile
COPY verl /root/myCodeLab/host/verl/
COPY Gym /root/myCodeLab/host/Gym/
COPY verifiable-instructions /tmp/verifiable-instructions/
```

Placed last: code change only invalidates L10-L15 (~2min rebuild), not L1-L4 (30min).

Directory names must match host filesystem so editable `.pth` paths resolve through the
docker-compose mount overlay.

### Layer 13 — Install verl + Gym + verifiable-instructions

```dockerfile
RUN cd /root/myCodeLab/host/verl && uv pip install --no-build-isolation --no-deps -e . && \
    cd /root/myCodeLab/host/Gym && uv pip install --no-build-isolation --no-deps -e . && \
    cd /tmp/verifiable-instructions && uv pip install --no-build-isolation --no-deps . && \
    rm -rf /tmp/verifiable-instructions
```

verl and Gym: editable (`-e`) — docker-compose mount replaces the COPY'd source with
live host source at runtime.

Gym is `--no-deps` to avoid pulling unwanted deps (mlflow, openai client, etc.).
Server infra deps (fastapi, uvicorn, aiohttp, ray) are already in the base image.

verifiable-instructions: non-editable (Gym dep for instruction_following domain).

### Layers 14-15 — Final

```dockerfile
WORKDIR /root/myCodeLab
CMD ["/bin/bash"]
```

Total new layers: 15 (107 + 15 = 122, under 127 limit).

---

## Docker Compose Section-by-Section Rationale

**File**: `docker-compose.yml`

### Mount Design

```yaml
volumes:
  - /mnt/public:/public                                          # Full shared fs
  - /mnt/public/lichang93/st_verl_dockerfile:/root/myCodeLab/host  # Project root
  - /mnt/public/lichang93/downloads:/root/myCodeLab/host/downloads  # Models + datasets
```

The project mount makes editable installs work at runtime — `.pth` files (in
`site-packages/`) point to `/root/myCodeLab/host/verl/` and `/root/myCodeLab/host/Gym/`
which resolve through the mount to live code.

Downloads mount is separate because `downloads/` lives outside the project dir on the
host (`/mnt/public/lichang93/downloads/`).

### Network & Security

| Setting | Value | Why |
|---------|-------|-----|
| `network_mode` | `host` | Ray/NCCL need direct host network; RDMA requires host NICs |
| `ipc` | `host` | PyTorch shared memory for GPU tensor transfers (default 64MB too small) |
| `privileged` | `true` | RDMA/InfiniBand `/dev/infiniband/` access |
| `memlock` | `-1` (unlimited) | RDMA pins GPU memory buffers |
| `stack` | `67108864` (64MB) | Deep call stacks in megatron-core pipeline parallelism |

### Environment Variables

**Core**: `CUDA_DEVICE_MAX_CONNECTIONS=1` — required by megatron-core pipeline parallelism.

**vLLM**: `VLLM_USE_V1=1`, `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` — B300 workaround:
vLLM's `is_device_capability(100)` does exact match but B300 reports sm_103. Forcing
CUTLASS_MLA selects block_size=128 codepath. Requires cuDNN >= 9.11 (image has 9.18).

**NCCL**:
- `NCCL_IB_HCA=mlx5_10:1,...,mlx5_17:1` — 8 RoCE NICs, one per GPU (PXB affinity)
- `NCCL_IB_GID_INDEX=3` — RoCEv2 routable GID (GID 0 is link-local, fails cross-node)
- `NCCL_IB_TC=138` — Traffic class matching cluster QoS config
- `NCCL_NVLS_ENABLE=1` — NVLink SHARP (NVSwitch in-network reductions)
- `NCCL_RUNTIME_CONNECT=0` — Establish connections at init, not lazily

**NVSHMEM/DeepEP**: `NVSHMEM_IBGDA_NIC_HANDLER=gpu`, `NVSHMEM_IB_ENABLE_IBGDA=1` —
GPU-initiated RDMA for MoE expert dispatch. TC=138 matches NCCL.

**Ray**: Disabled telemetry (no internet). `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` for
detecting hung NCCL collectives.

**wandb**: `WANDB_MODE=offline` — logs saved locally, sync later with `wandb sync`.

### Services

Both `head` and `worker` use `container_name: vrl` for consistent `docker exec` commands.

```yaml
# Head (b31): starts Ray head, binds 6379
ray start --head --port=6379 --dashboard-port=8265 --num-gpus=8 --num-cpus=64 --node-ip-address=10.12.11.5

# Worker (b32): joins head's cluster
ray start --address=10.12.11.5:6379 --num-gpus=8 --num-cpus=64 --node-ip-address=10.12.11.6
```

`restart: unless-stopped` — restart on daemon restart (`dvon`/`dvoff`), but not after
explicit stop or training crash (prevents checkpoint corruption).

---

## Iter2: Gym Server Integration

Gym resource servers run as local FastAPI services inside the container. Each domain
has its own `/verify` HTTP endpoint:

| Domain | Port | Data Source |
|--------|------|-------------|
| code_gen | 19001 | nemogym_code |
| mcqa | 19002 | nemogym_mcqa |
| instruction_following | 19003 | nemogym_if |
| structured_outputs | 19004 | nemogym_structured |
| workplace_assistant | 19005 | nemogym_workplace |
| math_with_judge | 19006 | nemogym_math |

**Key files**:
- `verl/my_scripts/gym_server_runner.py` — standalone launcher per domain
- `verl/my_scripts/launch_gym_servers.sh` — launches all 6 servers with health checks
- `verl/verl/workers/reward_manager/nemogym_server.py` — multi-domain reward manager
- `verl/my_scripts/run_moonlight_1node_blend_smoke.sh` — training script (auto-launches servers)

**Dataset**: `downloads/datasets/nemogym_blend/train_v2.parquet` — 93,244 samples across
all 6 domains (math patched from HF via `patch_blend_local.py`).

See `dev_manual_iter2_gym_server.md` for full implementation details and bug log.

---

## Hardware & Software Requirements

| Component | Version/Spec | Notes |
|-----------|-------------|-------|
| GPU | 8x B300 SXM6 per node (sm_103, 275GB HBM) | |
| Network | 8x 400Gb/s RoCE NICs (mlx5_10-17) per node | Per-GPU PXB affinity |
| cuDNN | >= 9.11 (image has 9.18) | MLA fused attention on sm >= 100 |
| Shared FS | /mnt/public (quarkfs) | Models, datasets, checkpoints |
| PyTorch | 2.10.0a0+nv25.11 | Base image (DO NOT MODIFY) |
| megatron-core | 0.16.0 | `/opt/Megatron-Bridge/3rdparty/Megatron-LM/` |
| transformer_engine | 2.12.0 | Base image |
| vLLM | 0.12.0 (custom build) | `/opt/vllm/` (editable, immutable at runtime) |
| deep_ep | 1.2.1 | Base image |
| flash_attn | 2.7.4.post1+25.11 | Base image |
| NeMo | 2.7.0 | `/opt/NeMo/` (not in pip metadata, but importable) |
| Ray | 2.54.0 | Base image |
| CUDA | 13.0 | Base image |
| NCCL | 2.28.9 | Base image |
| Python | 3.12.3 | Base image |

---

## Proxy Reference

```bash
von  / voff   # Shell proxy for pip/curl (defined in ~/.bashrc)
dvon / dvoff   # Docker daemon proxy (restarts docker!)
```

- `dvon` before `docker build` (pip needs internet)
- `dvoff` before `docker push` to Aliyun (can't reach registry through proxy)
- Proxy: `http://jdtcom:709a64b73eb3@10.119.176.202:3128`

---

## Related Docs

- `dev_manual_iter2_gym_server.md` — Iter2 migration plan, implementation details, bug log
- `image_deps_report_v2.md` — Exhaustive 3-image dependency diff (25.11.01 vs 26.02 vs myverl)
- `docs/pip_vs_uv_investigation.md` — pip vs uv dual-layer analysis, package shadowing
- `docs/reward_callstack_analysis.md` — verl reward manager callstack analysis
- `legacy/` — Old Dockerfiles, iter1 docs, MJ_NEMO_GYM, verl.old, inspection scripts
