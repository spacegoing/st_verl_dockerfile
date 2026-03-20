# README — verl Moonlight-16B Training Docker Image (nemo:26.02)

Images:
- `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base` — stable base (vLLM/CUDA)
- `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` — dev image (verl/Gym, FROM base)

Cluster: b31 (10.12.11.5) + b32 (10.12.11.6), 8x B300 SXM6 each

---

## Directory Structure

```
st_verl_dockerfile/
  Dockerfile.base               # Stable base: vLLM CUDA build + CUDA extensions + system deps
  Dockerfile.ncr.26.02.mydev   # Dev image: verl + Gym (FROM base, ~10 min rebuild, no CUDA)
  docker-compose.yml            # 2-node cluster orchestration (head + worker)
  readme.md                     # This file
  .gitignore / .dockerignore   # Git and Docker context config
  gym_config/                   # Gym server configs (COPYed to /opt/gym_config/ in image)
    gym_blend_servers.yaml      #   ng_run server config (ports 20001-20007, all domains)
    gym_env.yaml                #   policy_model placeholder env vars
  docker_assets/                # Static files COPYed into base image (tiktoken, tmux, nltk)
    o200k_base.tiktoken         #   tiktoken encoding (no internet at runtime)
    .tmux.conf / to_append.sh   #   shell / tmux config
    nltk_data/                  #   NLTK tokenization data
  docs/                         # All documentation
    readme.md (this file)
    dev_notes.md                #   Detailed changelog (all changes with rationale)
    dev_manual_iter2_gym_server.md  # Iter2 Gym server integration plan, bug log
    image_deps_report_v2.md     #   Exhaustive 3-image dependency diff
    iter3_standalone_pass_rate.md   # Iter3 pass-rate experiment notes
    reward_callstack_analysis.md    # verl reward manager callstack analysis
    pip_vs_uv_investigation.md  #   Why 26.02 has dual pip/uv, package shadowing analysis
    nemo_2602_image_anatomy.md  #   303-layer build phase analysis of nemo:26.02
  verl/                         # verl source (editable install)
    my_scripts/                 # Training launch scripts, Gym server launcher
  Gym/                          # NemoGym source (editable install; venvs live at /opt/gym_venvs/)
  vllm/                         # vLLM 0.12.0 source (built into base image, not editable at runtime)
  mbridge/                      # Megatron-Bridge pip package
  downloads/                    # Models + datasets (mount: /mnt/public/lichang93/downloads)
    models/Moonlight16B/        # Moonlight-16B weights
    datasets/nemogym_blend/     # Blend dataset (train_v2.parquet, val.parquet)
  gym_rollout/                  # Standalone pass-rate experiment (iter3)
  legacy/                       # Old Dockerfiles, iter1 docs, old scripts, MJ_NEMO_GYM/, verl.old/
```

---

## Quick Reference

### Build

Two-image strategy: rebuild `dev` (~10 min) for code changes; rebuild `base` (~30 min)
only when vLLM/CUDA/system packages change.

```bash
# ── Dev image rebuild (verl or Gym code changes) ─────────────────────────────
DOCKER_BUILDKIT=0 docker build -f Dockerfile.ncr.26.02.mydev \
  --network host \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev .
docker tag registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
           myverl:ncr2602_vllm012.dev
dvoff && docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev

# ── Base image rebuild (vLLM/CUDA/system package changes — rare) ──────────────
DOCKER_BUILDKIT=0 docker build -f Dockerfile.base \
  --network host \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base .
dvoff && docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base
# Then rebuild the dev image on top.
```

**Build times**:
- Dev image: ~10-15 min (no CUDA compilation; uv installs Gym venvs from cache in ~1-2 min)
- Base image: ~30 min (vLLM CUDA build); may need retries if cutlass git clone drops through proxy

**PyPI downloads use Aliyun mirror** (`https://mirrors.aliyun.com/pypi/simple/`) set via
`UV_DEFAULT_INDEX` in both Dockerfiles. This bypasses the corporate proxy for package downloads
(proxy throughput is ~50-100 KB/s; direct mainland access to Aliyun is much faster).
To override: `--build-arg UV_DEFAULT_INDEX=https://pypi.org/simple/`

**`DOCKER_BUILDKIT=0` required**: The base image has ~107 layers; the legacy builder's
127-layer limit demands consolidation (our 6 dev layers fit). BuildKit is not used because
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

Rebuild **dev image** when:
- Adding new Python packages to Gym or verl deps
- Gym dependency changes (ng_run dry_run recreates per-server venvs in `/opt/gym_venvs/`)
- Any change requiring fresh `/opt/gym_venvs/` venvs

Rebuild **base image** (rare) when:
- Changing vLLM version or vLLM source
- Adding/changing CUDA extension packages (`grouped_gemm`, `causal_conv1d`, `mamba_ssm`)
- Adding apt packages or changing `mbridge/`, static configs, or system Python deps

---

## Dockerfile Layer-by-Layer Rationale

Two-file split: `Dockerfile.base` (vLLM/CUDA, rebuilt rarely ~30 min) and
`Dockerfile.ncr.26.02.mydev` (verl/Gym on top of base, rebuilt frequently ~10 min).
This permanently avoids re-running vLLM CUDA compilation for code-only changes.

---

### Dockerfile.base — Stable base image

**Base**: `nvcr.io/nvidia/nemo:26.02` (~107 layers)

Layer order: slowest/most stable first (cache friendly).

#### Layer 1 — vLLM prep

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

#### Layer 2 — COPY vLLM source

```dockerfile
COPY vllm /opt/vllm
```

Source in `vllm/` must be pre-cleaned: delete `.git/`, `.deps/`, `tests/`, `docs/`,
`benchmarks/`, `examples/`, and all `*.so` files. Reduces COPY context from 3.8GB to ~26MB.

#### Layer 3 — vLLM CUDA build (~30 min)

```dockerfile
RUN cd /opt/vllm && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0 MAX_JOBS=64 NVCC_THREADS=1 \
    uv pip install --no-deps --no-build-isolation --no-cache -e .
```

Slowest step — placed in base image so it never runs again for code changes.

`--no-build-isolation`: Prevents creating an isolated venv that downloads its own
torch (CUDA 12.6), which would conflict with the base image's torch 2.10.0a0 (CUDA 13.0).

`--no-deps`: Protects all base image packages.

`MAX_JOBS=64 NVCC_THREADS=1`: Maximizes parallel C++ compilation; single-threaded NVCC
(memory-hungry, parallel threads cause OOM).

Clones `https://github.com/nvidia/cutlass.git` (~45MB) during cmake. May need retries if
proxy drops. Once built and pushed as base, this never runs again.

`-e .` (editable): `/opt/vllm` is NOT under the docker-compose mount, so vLLM always
resolves from the image layer (immutable). This is intentional.

#### Layer 4 — CUDA extension packages

```dockerfile
RUN uv pip install --no-cache --no-deps --no-build-isolation grouped_gemm && \
    uv pip install --no-cache --no-deps --no-build-isolation causal_conv1d && \
    uv pip install --no-cache --no-deps --no-build-isolation mamba_ssm
```

Were in nemo:25.11.01 but removed in 26.02. Imported by megatron-core (MoE expert GEMM,
SSM architecture). Each requires NVCC compilation.

#### Layer 5 — System packages + pure-Python deps

```dockerfile
RUN apt-get update && apt-get install -y pdsh tmux htop vim && \
    rm -rf /var/lib/apt/lists/* && \
    git config --global --add safe.directory '*' && \
    uv pip install --no-cache --no-deps \
        wandb gpustat codetiming tensordict mathruler pylatexenc torchdata \
        hydra-core bitsandbytes orjson \
        transformers==4.57.3 \
        pyvers \
        yappi itsdangerous gprof2dot pydot && \
    wandb login df3cecbfc0874c8a352c40820becf4a15575614e
```

All pure-Python or pre-built wheels — no NVCC needed.

`uv pip install` writes to `/opt/venv/lib/python3.12/site-packages/` (HIGH priority).
This is critical — plain `pip install` would write to `/usr/local/` (LOW priority),
where packages are shadowed by the venv copy. See `docs/pip_vs_uv_investigation.md`.

`--no-deps` for every install: Without it, `wandb` would pull `numpy>=2.0` which
would overwrite the base's `numpy==1.26.4`, breaking every CUDA extension's C ABI.

`yappi itsdangerous gprof2dot pydot`: Required by Gym's `profiling.py` module-level imports.

`git config --global --add safe.directory '*'`: Prevents "dubious ownership" errors in
mounted volumes.

#### Layer 6 — mbridge

```dockerfile
COPY mbridge /tmp/mbridge
RUN cd /tmp/mbridge && uv pip install --no-cache --no-deps . && rm -rf /tmp/mbridge
```

Pure-Python bridge between verl and megatron-core. Non-editable (rarely changes).

#### Layer 7 — Static config files

```dockerfile
COPY o200k_base.tiktoken to_append.sh .tmux.conf /tmp/docker_context/
COPY nltk_data /usr/local/share/nltk_data
```

`o200k_base.tiktoken`: vLLM 0.12 uses OpenAI's tiktoken encoding (no internet at runtime).
`nltk_data/`: NLTK tokenization data used by Gym's math evaluation.

#### Layer 8 — Environment variables

```dockerfile
ENV NLTK_DATA=/usr/local/share/nltk_data \
    TIKTOKEN_ENCODINGS_BASE=/root/tiktoken_cache
```

#### Layer 9 — Workspace setup

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

---

### Dockerfile.ncr.26.02.mydev — Dev image (FROM base)

**Base**: `ncr2602_vllm012.base` (above, ~122 layers)
**Rebuild time**: ~10-15 min, no CUDA compilation.

#### Layers 10-11 — COPY codebases (change most often)

```dockerfile
COPY verl /root/myCodeLab/host/verl/
COPY Gym /root/myCodeLab/host/Gym/
```

Placed last: code change only invalidates L10-L15 (~10 min rebuild), not base (30 min).

`verifiable-instructions` no longer COPYed — Gym's `lc_fix` branch pulls it from
`git+https://github.com/spacegoing/my_verifiable-instructions.git` via
`instruction_following/requirements.txt`.

#### Layer 12 — Install verl

```dockerfile
RUN cd /root/myCodeLab/host/verl && uv pip install --no-build-isolation --no-deps -e .
```

verl editable install into `/opt/venv/` — accessible to the training loop.

#### Layer 13 — Gym isolated venv + ng_run per-server venvs

```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cd /root/myCodeLab/host/Gym && \
    mkdir -p cache && \
    /root/.local/bin/uv venv --python 3.12 /opt/gym_venvs/main && \
    /root/.local/bin/uv pip install -e ".[dev]" --python /opt/gym_venvs/main/bin/python && \
    /root/.local/bin/uv pip install \
        langdetect openapi_schema_validator math_verify \
        absl-py nltk immutabledict \
        --python /opt/gym_venvs/main/bin/python && \
    cp /opt/gym_config/gym_env.yaml /root/myCodeLab/host/Gym/env.yaml && \
    /opt/gym_venvs/main/bin/ng_run \
        "+config_paths=[/opt/gym_config/gym_blend_servers.yaml]" \
        "+dry_run=true" \
        "+uv_venv_dir=/opt/gym_venvs"
```

Gym venvs are placed at `/opt/gym_venvs/` (NOT under `/root/myCodeLab/host/`). Key reasons:
- **Mount override**: docker-compose mounts the PFS project dir onto `/root/myCodeLab/host/`,
  which would wipe any venvs baked there. `/opt/` survives the mount.
- **PFS slowness**: quarkfs is slow for Python's small random file I/O. `/opt/` is
  node-local disk — Python import is ~5-10x faster than on the shared filesystem.

Key steps:

1. **uv upgrade** — Gym requires >= 0.9.30 (base has 0.7.2); installed to `~/.local/bin/`
2. **Gym main venv** at `/opt/gym_venvs/main/` — `.[dev]` + domain verifier packages
3. **ng_run `+dry_run=true` `+uv_venv_dir=/opt/gym_venvs`** — creates per-server venvs at
   `/opt/gym_venvs/resources_servers/<name>/.venv/` (~850-978ms each on overlay, uv cache warm).
   At runtime, `skip_venv_if_present=true` detects existing venvs → 18s startup (no reinstall)
4. **env.yaml** — policy_model placeholder for `${policy_base_url}` interpolation; copied from
   `/opt/gym_config/` (self-contained — decoupled from `verl/my_scripts/`)
5. **lcb_integration (code_gen Ray workers)** — `lcb_integration` has no `pyproject.toml`
   and cannot be pip-installed. Ray workers spawn in a temp dir, not `code_gen/`, so
   `import lcb_integration` fails without explicit path setup. Fix is in `compute_code_generation_metrics.py`:
   ```python
   _CODE_GEN_DIR = str(Path(__file__).parent.parent)  # → code_gen/ at import time
   @ray.remote(runtime_env={
       "py_executable": sys.executable,          # use code_gen venv, not system Python
       "env_vars": {"PYTHONPATH": _CODE_GEN_DIR}, # so workers can find lcb_integration
   })
   ```
   No symlink or Dockerfile change needed — the fix is entirely in Python.

See `verl/plans/nemo_gym_worker/` for full design docs.

#### Layers 14-15 — Final

```dockerfile
WORKDIR /root/myCodeLab
CMD ["/bin/bash"]
```

Total layers: base ~122 + 6 dev layers = ~128 (near 127 limit — merge COPY layers if needed).

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
| `nofile` | `65536` | Gym Ray cluster with `--num-cpus=32` needs >1024 fds |

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

## Gym Server Integration (ng_run)

Gym resource servers run as local FastAPI services via `ng_run` (Gym's official CLI).
Each domain has its own isolated `.venv` and `/verify` HTTP endpoint:

| Domain | Port | Data Source | Uses Ray? |
|--------|------|-------------|-----------|
| code_gen | 20001 | nemogym_code | YES |
| mcqa | 20002 | nemogym_mcqa | No |
| instruction_following | 20003 | nemogym_if | No |
| structured_outputs | 20004 | nemogym_structured | No |
| workplace_assistant | 20005 | nemogym_workplace | No |
| math_with_judge | 20006 | nemogym_math | No |

**Architecture**: Dual Ray cluster design — Gym Ray (port 6380, `/tmp/ray_gym/`) isolated
from verl Ray (port 6379, `/tmp/ray/`). See `verl/plans/nemo_gym_worker/design_manual.md`.

**Venv locations** (runtime):
- Gym main venv: `/opt/gym_venvs/main/` (Ray, ng_run binary)
- Per-server venvs: `/opt/gym_venvs/resources_servers/<name>/.venv/` (7 servers)
- All venvs are on node-local disk (`/opt/`), NOT on PFS, and survive docker-compose mounts

**Key files**:
- `verl/my_scripts/start_gym_uv.sh` — starts Gym Ray cluster + all 6 servers via ng_run
- `verl/my_scripts/gym_blend_servers.yaml` — server config (ports 20001-20006)
- `verl/my_scripts/gym_env.yaml` — policy_model placeholder env vars
- `verl/verl/workers/reward_manager/nemogym_server.py` — multi-domain reward manager
- `verl/my_scripts/run_moonlight_1node_blend_smoke.sh` — training script (auto-launches servers)
- `verl/plans/nemo_gym_worker/` — full design docs, research, configs

**Dataset**: `downloads/datasets/nemogym_blend/train_v2.parquet` — 93,244 samples across
all 6 domains (math patched from HF via `patch_blend_local.py`).

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
- `dev_notes.md` — Detailed changelog: pip→uv migration, build results, all changes with rationale
- `docs/pip_vs_uv_investigation.md` — pip vs uv dual-layer analysis, package shadowing
- `docs/nemo_2602_image_anatomy.md` — 303-layer build phase analysis (what NeMo adds vs PyTorch base)
- `docs/reward_callstack_analysis.md` — verl reward manager callstack analysis
- `legacy/` — Old Dockerfiles, iter1 docs, MJ_NEMO_GYM, verl.old, inspection scripts
