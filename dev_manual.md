# Dev Manual: verl Moonlight-16B 2-Node B300 Training

**Started**: 2026-03-10
**Image**: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`
**Cluster**: b31 (10.12.11.5) + b32 (10.12.11.6), 8× B300 SXM6 each (sm_103)

---

## Bugs & Fixes Log

| # | Date | Phase | Issue | Root Cause | Fix |
|---|------|-------|-------|------------|-----|
| 1 | 03-10 | 0 | `/root/myCodeLab/host` symlink broken — verl/mjnemogym source dirs not found inside container | Dockerfile creates symlink `host -> /public/lichang93/stCodeLab` but `stCodeLab` didn't exist on this new cluster's shared fs | Created `/mnt/public/lichang93/stCodeLab/` with subdirs; symlinked `verl` and `mjnemogym` into it using **container-visible** paths (`/public/...` not `/mnt/public/...`) |
| 2 | 03-10 | 0 | Dataset download 401 error for `BytedTsinghua-SIA/DAPO-Math-17k` and `AIME-2024` | Used `huggingface-cli download` without `--repo-type dataset` (defaults to model type) | Added `--repo-type dataset` flag |
| 3 | 03-10 | 0 | Model download `FileExistsError: /root/myCodeLab/host` | `mkdir -p` tried to create dirs along the symlink path, but `/root/myCodeLab/host` is a symlink to non-existent target | Fixed by creating the target dir structure first (Bug #1 fix resolved this too) |
| 4 | 03-10 | 1 | `ModuleNotFoundError: No module named 'verl'` | Editable install in image pointed to COPY'd path `/root/myCodeLab/host/verl/`, but at runtime that resolves through symlink to shared fs which was empty | After Bug #1 fix, re-ran `pip install --no-deps -e .` from the symlinked verl source |
| 5 | 03-10 | 1 | `ModuleNotFoundError: No module named 'pyvers'` | `tensordict` depends on `pyvers` but it was installed with `--no-deps` in Dockerfile | `pip install --no-cache-dir --no-deps pyvers` — **must add to Dockerfile** |
| 6 | 03-10 | 1 | `ModuleNotFoundError: No module named 'math_verify'` | `mjnemogym` depends on `math_verify` but was installed with `--no-deps` | `pip install --no-cache-dir --no-deps math_verify` — **must add to Dockerfile** |
| 7 | 03-10 | 1 | `ModuleNotFoundError: No module named 'latex2sympy2_extended'` | `math_verify` depends on `latex2sympy2_extended` | `pip install --no-cache-dir --no-deps latex2sympy2_extended` — **must add to Dockerfile** |
| 8 | 03-10 | 1 | `ModuleNotFoundError: No module named 'openapi_schema_validator'` | `mjnemogym` structured_outputs module depends on it | `pip install --no-cache-dir --no-deps openapi_schema_validator` — **must add to Dockerfile** |
| 9 | 03-10 | 2 | `ModuleNotFoundError: No module named 'orjson'` | `verl.utils.tracking` imports `orjson` | `pip install --no-cache-dir --no-deps orjson` — **must add to Dockerfile** |
| 10 | 03-10 | 2 | Megatron: 27 layers not divisible by PP=2 | `num_layers % pipeline_model_parallel_size == 0` assertion | Changed `train_pp=1` (was 2) in `run_moonlight_2node_test.sh` |
| 11 | 03-10 | 2 | VPP requires PP>1: `RuntimeError: pipeline-model-parallel size should be greater than 1 with interleaved schedule` | VPP (virtual pipeline parallelism) is incompatible with PP=1 | Set `vpp=null` (was 2) in `run_moonlight_2node_test.sh` |
| 12 | 03-10 | 2 | NCCL connects to wrong IP (10.125.0.203 via bond0.200 VLAN) | `NCCL_SOCKET_IFNAME=bond0` prefix-matches `bond0.200` VLAN interface. NCCL NET/IB plugin couldn't find IB devices because container lacked `/dev/infiniband/` | Use `privileged: true` in docker-compose (not just `--device=/dev/infiniband/`) so containers have full IB/RDMA access |
| 13 | 03-10 | 2 | Docker compose `command:` YAML folded scalar broke multi-line ray start command (`--num-gpus=8: command not found`) | YAML `>` folded scalar split the command at indentation boundaries | Changed to list format: `command: ["-c", ">-\n ray start ... && tail -f /dev/null"]` |
| 14 | 03-10 | 2 | NCCL IB error: `status=12 vendor err 129` — RoCE packets not routable between nodes | `NCCL_IB_GID_INDEX=0` selects link-local GID (`fe80::`) which isn't routable across nodes. Need RoCEv2 IPv4-mapped GID | Changed `NCCL_IB_GID_INDEX=3` (RoCEv2, IPv4-mapped `::ffff:100.86.0.x`) in both `docker-compose.yml` and `my_deepep_env.yaml` |
| 15 | 03-10 | 3 | vLLM KV cache init: `ValueError: No common block size for 16` | **vLLM bug**: `is_device_capability(100)` does exact match, but B300 is sm_103 (10.3 ≠ 10.0). Blackwell detection fails → falls back to FlashMLA Dense which is Hopper-only → block_size=16 stays, incompatible with MLA backends | Set `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` in `my_deepep_env.yaml` and `docker-compose.yml` (forces block_size=128) |

---

## Dockerfile Patches Needed

These packages must be added to the Dockerfile Layer 1 (or a new layer):
```dockerfile
pip install --no-cache-dir --no-deps pyvers math_verify latex2sympy2_extended openapi_schema_validator orjson
```

---

## Runtime Setup Required (Not in Image)

On every fresh container start on this cluster, the following is needed:

1. **Create `/mnt/public/lichang93/stCodeLab/` structure** (one-time, on shared fs):
   ```bash
   mkdir -p /mnt/public/lichang93/stCodeLab/downloads/{models,datasets/dapo_data}
   ln -sf /public/lichang93/st_verl_dockerfile/verl /mnt/public/lichang93/stCodeLab/verl
   ln -sf /public/lichang93/st_verl_dockerfile/MJ_NEMO_GYM /mnt/public/lichang93/stCodeLab/mjnemogym
   ```
   **Important**: Symlinks must use `/public/...` (container path), not `/mnt/public/...` (host path).

2. **Install missing pip deps** (once per container):
   ```bash
   pip install --no-cache-dir --no-deps pyvers math_verify latex2sympy2_extended openapi_schema_validator orjson
   ```

3. **Re-install editable packages** (once per container, since source is on shared fs):
   ```bash
   cd /root/myCodeLab/host/verl && pip3 install --no-deps -e .
   cd /root/myCodeLab/host/mjnemogym && pip3 install --no-deps -e .
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
| GPU detection (8× B300) | PASS |
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
| GPU memory | 208.5 GB / 275 GB per GPU |
| Throughput | 7.12 tokens/s |

---

## Cluster Info

- GPU: NVIDIA B300 SXM6, sm_103, 275GB HBM each
- cuDNN: 9.18 (>= 9.11 required for MLA fused attention)
- GPU-NIC affinity: GPU0↔mlx5_10, GPU1↔mlx5_11, ..., GPU7↔mlx5_17 (PXB topology)
- IB GID table: GID0=fe80:: (link-local), GID3=::ffff:100.86.0.x (RoCEv2, routable)
- IB devices: mlx5_z0–z3 (InfiniBand), mlx5_10–17 (RoCE 400Gb/s)
- Shared fs: `/mnt/public` (quarkfs, 1PB) mounted on both nodes
- Proxy: `http://jdtcom:709a64b73eb3@10.119.176.202:3128` (in ~/.bashrc as `dvon`/`dvoff`)
