# Dev Manual: verl Moonlight-16B 2-Node B300 Training

**Started**: 2026-03-10
**Image**: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`
**Cluster**: b31 (10.12.11.5) + b32 (10.12.11.6), 8x B300 SXM6 each (sm_103)

---

## Bugs & Fixes Log

| # | Date | Phase | Issue | Root Cause | Fix |
|---|------|-------|-------|------------|-----|
| 1 | 03-10 | 0 | `/root/myCodeLab/host` symlink broken — verl/mjnemogym source dirs not found inside container | Dockerfile creates symlink `host -> /public/lichang93/stCodeLab` but `stCodeLab` didn't exist on this new cluster's shared fs | **Superseded by v2 Dockerfile**: removed all symlinks; mount `st_verl_dockerfile` directly as `/root/myCodeLab/host` in docker-compose |
| 2 | 03-10 | 0 | Dataset download 401 error for `BytedTsinghua-SIA/DAPO-Math-17k` and `AIME-2024` | Used `huggingface-cli download` without `--repo-type dataset` (defaults to model type) | Added `--repo-type dataset` flag |
| 3 | 03-10 | 0 | Model download `FileExistsError: /root/myCodeLab/host` | `mkdir -p` tried to create dirs along the symlink path, but `/root/myCodeLab/host` is a symlink to non-existent target | **Superseded by v2 Dockerfile**: `host` is now a plain dir (mount point), not a symlink |
| 4 | 03-10 | 1 | `ModuleNotFoundError: No module named 'verl'` | Editable install in image pointed to COPY'd path, but at runtime symlink resolved to empty shared fs | **Superseded by v2 Dockerfile**: mount overlays COPY'd source; editable install points to mount |
| 5 | 03-10 | 1 | `ModuleNotFoundError: No module named 'pyvers'` | `tensordict` depends on `pyvers` but it was installed with `--no-deps` in Dockerfile | Added `pyvers` to Dockerfile Layer 5 |
| 6 | 03-10 | 1 | `ModuleNotFoundError: No module named 'math_verify'` | `mjnemogym` depends on `math_verify` but was installed with `--no-deps` | Added `math_verify` to Dockerfile Layer 5 |
| 7 | 03-10 | 1 | `ModuleNotFoundError: No module named 'latex2sympy2_extended'` | `math_verify` depends on `latex2sympy2_extended` | Added `latex2sympy2_extended` to Dockerfile Layer 5 |
| 8 | 03-10 | 1 | `ModuleNotFoundError: No module named 'openapi_schema_validator'` | `mjnemogym` structured_outputs module depends on it | Added `openapi_schema_validator` to Dockerfile Layer 5 |
| 9 | 03-10 | 2 | `ModuleNotFoundError: No module named 'orjson'` | `verl.utils.tracking` imports `orjson` | Added `orjson` to Dockerfile Layer 5 |
| 10 | 03-10 | 2 | Megatron: 27 layers not divisible by PP=2 | `num_layers % pipeline_model_parallel_size == 0` assertion | Changed `train_pp=1` (was 2) in `run_moonlight_2node_test.sh` |
| 11 | 03-10 | 2 | VPP requires PP>1: `RuntimeError` | VPP (virtual pipeline parallelism) is incompatible with PP=1 | Set `vpp=null` (was 2) in `run_moonlight_2node_test.sh` |
| 12 | 03-10 | 2 | NCCL connects to wrong IP (10.125.0.203 via bond0.200 VLAN) | `NCCL_SOCKET_IFNAME=bond0` prefix-matches `bond0.200` VLAN interface. Container lacked `/dev/infiniband/` | Use `privileged: true` in docker-compose for full IB/RDMA access |
| 13 | 03-10 | 2 | Docker compose `command:` YAML folded scalar broke ray start | YAML `>` folded scalar split command at indentation boundaries | Changed to list format: `command: ["-c", ">- ..."]` |
| 14 | 03-10 | 2 | NCCL IB error: `status=12 vendor err 129` — RoCE not routable | `NCCL_IB_GID_INDEX=0` selects link-local GID (`fe80::`) | Changed to `NCCL_IB_GID_INDEX=3` (RoCEv2, `::ffff:100.86.0.x`) |
| 15 | 03-10 | 3 | vLLM KV cache: `ValueError: No common block size for 16` | **vLLM bug**: `is_device_capability(100)` exact match fails for B300 sm_103. Falls back to FlashMLA Dense (Hopper-only) | Set `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` (forces block_size=128). cuDNN >= 9.11 required for MLA fused attn on sm >= 100; image has 9.18 |
| 16 | 03-11 | img | b31 and b32 had different image IDs (b58dc vs 9b98e) | b31 built locally (shares base layers, shows 52.8GB), b32 loaded from tar (no shared layers, shows 107GB). Functionally equivalent but not identical | **v2 rebuild**: single image, push to registry, pull on both nodes |
| 17 | 03-11 | img | Dockerfile used symlinks to stCodeLab (fragile, cluster-specific) | Image assumed host path `/public/lichang93/stCodeLab` exists | **v2 Dockerfile**: removed all symlinks; `/root/myCodeLab/host` is a plain dir (mount point set in compose) |

---

## Dockerfile v2 Design (2026-03-11)

### Design Principles
1. **Heavy compilation first** (vLLM CUDA build, CUDA extensions) — maximizes layer cache hits
2. **Frequently-changed code last** (verl, mjnemogym) — fast rebuilds on code changes
3. **All `pip install --no-deps`** — protects base image packages from being modified
4. **No host path assumptions** — `/root/myCodeLab/host` is a plain empty dir in image; compose mounts the actual project root
5. **No symlink workarounds** — mount path in compose is the single source of truth

### Layer Order (14 layers, 121 total with base)
```
L1:  pip uninstall vllm + install setuptools_scm     (prep)
L2:  COPY vllm source                                (rarely changes)
L3:  RUN vllm build                                  (SLOWEST ~30min, cached)
L4:  RUN grouped_gemm + causal_conv1d + mamba_ssm    (CUDA compile, cached)
L5:  RUN apt + pip pure-python deps                  (rarely changes)
L6:  COPY+RUN mbridge                                (small, stable)
L7:  COPY configs (tiktoken, nltk, bashrc, tmux)     (stable)
L8:  ENV                                             (stable)
L9:  RUN workspace setup                             (stable)
L10: COPY verl                                       (changes often)
L11: COPY mjnemogym                                  (changes often)
L12: RUN pip install -e verl + mjnemogym             (depends on L10-L11)
L13: WORKDIR
L14: CMD
```

### Mount Design (docker-compose.yml)
```yaml
volumes:
  - /mnt/public:/public                                          # full shared fs
  - /mnt/public/lichang93/st_verl_dockerfile:/root/myCodeLab/host  # project root
```
- Inside container: `/root/myCodeLab/host/verl/`, `/root/myCodeLab/host/MJ_NEMO_GYM/`, `/root/myCodeLab/host/downloads/`
- Downloads symlinked: `st_verl_dockerfile/downloads -> stCodeLab/downloads` (host-side, 30GB model data)
- No runtime `pip install -e` needed — editable install in image works because mount overlays the same path

### Build & Deploy
```bash
# On b31 (build node):
dvon                                          # docker proxy ON for pip
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.ncr.26.02.mydev \
  --network host \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev .
dvoff                                         # proxy OFF for Aliyun push
docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev

# On b32 (pull):
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
```

---

## Runtime Setup (v2 — after rebuild)

With the v2 image + updated compose, **no manual setup is needed** for verl/mjnemogym/pip deps.

Only needed once on host (already done):
```bash
# Downloads symlink (host-side, one-time):
ln -sf /mnt/public/lichang93/stCodeLab/downloads /mnt/public/lichang93/st_verl_dockerfile/downloads
```

---

## Key Config Changes (from original 64-node script)

| Parameter | Original (64-node) | 2-Node B300 | Reason |
|-----------|-------------------|-------------|--------|
| `train_pp` | 2 | 1 | 27 layers not divisible by 2 |
| `vpp` | 2 | null | VPP requires PP>1 |
| `EP` | 8 | 8 | Same (8 experts per EP group) |
| `NNODES` | 64 | 2 | Cluster size |
| `NCCL_IB_GID_INDEX` | 0 | 3 | B300 RoCE needs RoCEv2 GID |
| `VLLM_ATTENTION_BACKEND` | (not set) | CUTLASS_MLA | B300 sm_103 not detected as Blackwell by vLLM |

---

## Changelog

### Phase 0: Prerequisites — DONE

| Task | Status |
|------|--------|
| /etc/hosts on both nodes | Done |
| Moonlight-16B download (27 shards, ~30GB) | Done |
| DAPO-Math-17k dataset | Done (286MB, 1.79M rows) |
| AIME-2024 dataset | Done (29KB, 960 rows) |
| Docker image on both nodes | Done |
| pdsh installed on b31 | Done |

### Phase 1: Single-Container Smoke Tests — DONE

| Test | Status |
|------|--------|
| All imports (torch, vllm, verl, megatron-core, TE, mbridge, deep_ep, etc.) | PASS |
| GPU detection (8x B300) | PASS |
| Dataset schema verification | PASS |

### Phase 2: Ray Cluster — DONE

| Test | Status |
|------|--------|
| Ray head on b31 | PASS |
| Ray worker on b32 | PASS |
| 16 GPUs, 128 CPUs visible | PASS |
| IB devices accessible (`/dev/infiniband/`) | PASS (privileged mode) |

### Phase 3: Training — PASS

| Test | Status |
|------|--------|
| NCCL cross-node communication | PASS (GID_INDEX=3, RoCEv2) |
| Model weight loading across 2 nodes | PASS (~20min from shared fs) |
| vLLM rollout init (CUTLASS_MLA) | PASS |
| Validation before train (step:0) | PASS |
| Training step 1 | PASS (pg_loss=0.79, grad_norm=6.1, 796s/step) |
| Training step 2 | PASS (pg_loss=0.75, grad_norm=6.3, 809s/step) |
| GPU memory | 209 GB / 275 GB per GPU |
| Throughput | 6.4-7.1 tokens/s |

### Phase 4: Dockerfile v2 Redesign — 2026-03-11

| Task | Status |
|------|--------|
| Dockerfile holistic redesign (heavy layers first, no symlinks) | Done |
| docker-compose.yml mount path fix (no stCodeLab indirection) | Done |
| dev_manual.md updated | Done |
| Downloads symlink on host fs | Done |
| Image rebuild | Done (88e453cd896f, b32) |
| Push to Aliyun registry | Pending (need `dvoff` first) |
| Dockerfile fix: `mjnemogym/` → `MJ_NEMO_GYM/` | Done |
| Dockerfile fix: `--no-build-isolation` for editable installs | Done |
| docker-compose: added downloads mount | Done |
| Removed stale stCodeLab symlink | Done |
| Smoke test: imports, model load, vLLM CUTLASS_MLA init | PASS |
| Smoke test: training step | PASS (reward manager `data_source` mismatch — verl code issue, not image) |
| Note: `enforce_eager=True` needed (torch.compile shape bug on B300) | Bug #18 |

---

## Cluster Info

- GPU: NVIDIA B300 SXM6, sm_103, 275GB HBM each
- cuDNN: 9.18 (>= 9.11 required for MLA fused attention on sm >= 100)
- GPU-NIC affinity: GPU0<->mlx5_10, GPU1<->mlx5_11, ..., GPU7<->mlx5_17 (PXB topology)
- IB GID table: GID0=fe80:: (link-local), GID3=::ffff:100.86.0.x (RoCEv2, routable)
- IB devices: mlx5_z0-z3 (InfiniBand), mlx5_10-17 (RoCE 400Gb/s)
- Shared fs: `/mnt/public` (quarkfs, 1PB) mounted on both nodes
- Proxy: `von`/`voff` (shell), `dvon`/`dvoff` (docker daemon+client)
  - `dvon` enables proxy for `docker build` (pip needs internet)
  - `dvoff` needed before `docker push` to Aliyun (registry unreachable via proxy)
