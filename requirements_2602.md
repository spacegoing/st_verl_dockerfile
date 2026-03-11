# requirements_2602.md — Exhaustive Dependency & Change List

All changes required on top of `nvcr.io/nvidia/nemo:26.02` to run Moonlight-16B
GRPO training via `verl/my_scripts/run_moonlight_math_megatron_h200.sh`.

---

## Part A: verl Call Stack Runtime Dependencies

Everything the verl training pipeline imports or loads at runtime.
Organized by the call chain that discovered each need.

### A1. vLLM 0.12.0 (replaces base image's 0.14.2)

The 26.02 base ships vLLM 0.14.2. verl requires exactly 0.12.0 (API changes
in 0.13+ break verl's rollout worker integration).

| Action | Detail |
|--------|--------|
| Uninstall base vLLM | `pip uninstall -y vllm` (in 26.02, vLLM is a pip package, no `/opt/vllm/` dir) |
| Install `setuptools_scm` | vLLM build system needs it; `.git` stripped from source so also need `SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0` |
| Build vLLM 0.12.0 from source | `pip install --no-deps --no-build-isolation -e .` with `MAX_JOBS=64 NVCC_THREADS=1` |
| Build flags | `--no-build-isolation` required: without it, pip downloads its own torch (CUDA 12.6) conflicting with system CUDA 13.0. `--no-deps` protects base packages |

**Source preparation**: vLLM source at `vllm/` must be cleaned before COPY:
- Delete `.git/`, `.deps/` (1.6GB build cache), `tests/`, `docs/`, `benchmarks/`, `examples/`
- Delete all `*.so` files (compiled against old PyTorch 2.9, will be rebuilt)
- Reduces build context from 3.8GB to ~26MB

### A2. CUDA Extension Packages (removed from 26.02, were in 25.11.01)

These are imported by megatron-core and the model architecture. All require
NVCC compilation and must use `--no-build-isolation` to use system torch.

| Package | Version | Import Chain | Why Removed from 26.02 |
|---------|---------|-------------|----------------------|
| `grouped_gemm` | 0.3.0 | `megatron.core` MoE expert GEMM | Packaging cleanup in 26.02 |
| `causal_conv1d` | 1.6.0 | `mamba_ssm` → Mamba/SSM architecture | Packaging cleanup in 26.02 |
| `mamba_ssm` | 2.3.0 | `megatron.core` SSM support | Packaging cleanup in 26.02 |

Install: `pip install --no-cache-dir --no-deps --no-build-isolation <pkg>` for each.

### A3. Pure-Python Packages Missing from 26.02

Discovered via `ModuleNotFoundError` at runtime. All installed with `--no-deps`
to avoid pulling transitive deps that could corrupt base packages.

| Package | Version | Import Chain | Bug # |
|---------|---------|-------------|-------|
| `wandb` | 0.25.0 | `verl.trainer` → `wandb` (logging) | — (not in 26.02, was in 25.11.01) |
| `hydra-core` | 1.3.2 | `verl.trainer.main_ppo` → `hydra` (config management) | — (not in 26.02, was in 25.11.01) |
| `orjson` | 3.11.7 | `verl.utils.tracking` → `orjson` | Bug #9 |
| `pyvers` | 0.2.2 | `tensordict` → `pyvers` | Bug #5 |
| `math_verify` | 0.9.0 | `mjnemogym.math_with_judge` → `math_verify` | Bug #6 |
| `latex2sympy2_extended` | 1.11.0 | `math_verify` → `latex2sympy2_extended` | Bug #7 |
| `openapi_schema_validator` | 0.8.1 | `mjnemogym.structured_outputs` → `openapi_schema_validator` | Bug #8 |
| `bitsandbytes` | 0.49.2 | `verl` quantization support | — (not in 26.02, was in 25.11.01) |

### A4. Packages Already in 26.02 But Need Version Override

| Package | 26.02 Version | Required Version | Why |
|---------|--------------|-----------------|-----|
| `transformers` | 4.56.0 | **4.57.3** | verl + Moonlight model config need features from 4.57+ |

Install: `pip install --no-cache-dir --no-deps transformers==4.57.3`

### A5. verl Direct Dependencies (from `setup.py`)

These are verl's declared `install_requires`. Most are in the 26.02 base already.

| Package | Constraint | In 26.02? | Action |
|---------|-----------|-----------|--------|
| `accelerate` | any | Yes (1.12.0) | None |
| `codetiming` | any | **No** | Install |
| `datasets` | any | Yes (3.1.0) | None |
| `dill` | any | Yes (0.3.8) | None |
| `hydra-core` | any | **No** | Install (see A3) |
| `numpy` | <2.0.0 | Yes (1.26.4) | None (satisfies) |
| `pandas` | any | Yes (2.3.3) | None |
| `peft` | any | Yes (0.18.1) | None |
| `pyarrow` | >=19.0.0 | Yes (22.0.0) | None |
| `pybind11` | any | Yes (3.0.1) | None |
| `pylatexenc` | any | **No** | Install |
| `ray[default]` | >=2.41.0 | Yes (2.54.0) | None |
| `torchdata` | any | **No** | Install |
| `tensordict` | >=0.8.0,<=0.10.0,!=0.9.0 | Yes (0.11.0) | **Note**: 0.11.0 exceeds constraint; works at runtime |
| `transformers` | any | Yes (4.56.0) | Override to 4.57.3 (see A4) |
| `wandb` | any | **No** | Install (see A3) |
| `packaging` | >=20.0 | Yes (25.0) | None |
| `tensorboard` | any | Yes (2.20.0) | None |

verl optional extras used in our config:
- `[math]`: needs `math-verify` → install
- `[mcore]`: needs `mbridge` → install from local source
- `[vllm]`: needs `vllm>=0.8.5,<=0.12.0` → building from source

### A6. mjnemogym Dependencies

Key deps from `MJ_NEMO_GYM/pyproject.toml` that are NOT in 26.02:

| Package | Constraint | In 26.02? | Action |
|---------|-----------|-----------|--------|
| `math-verify` | >=0.8.0 | **No** | Install (see A3) |
| `latex2sympy2_extended` | >=1.10.2 | **No** | Install (see A3) |
| `openapi-schema-validator` | >=0.6.3 | **No** | Install (see A3) |
| `nltk` | >=3.9.2 | Yes | None (but needs data files — `nltk_data/` COPY) |
| `spacy` | any | Yes | None |
| `antlr4-python3-runtime` | ==4.9.3 | Yes | None |

Most other mjnemogym deps (`sympy`, `pandas`, `jsonschema`, `tqdm`, `aiohttp`,
etc.) are already in the 26.02 base.

### A7. mbridge

`mbridge` (v0.15.1) has **zero declared dependencies** in its pyproject.toml.
It's a pure-Python bridge between verl and megatron-core. Install with
`--no-deps` from local COPY.

### A8. Other Utility Packages

| Package | Why Needed | In 26.02? |
|---------|-----------|-----------|
| `gpustat` | GPU monitoring (`gpustat -cup`) | **No** |
| `tensordict` | verl data structures | Yes (0.11.0) |
| `mathruler` | Math evaluation | **No** |
| `torchdata` | Data loading | **No** |
| `model_hosting_container_standards` | vLLM dep (may be orphaned after uninstall) | Yes (0.1.13) |
| `anthropic` | vLLM dep (may be orphaned after uninstall) | Yes (0.71.0) |

### A9. Static Data Files Required at Runtime

| File | Container Path | Why |
|------|---------------|-----|
| `nltk_data/` (tokenizers) | `/usr/local/share/nltk_data` | NLTK tokenization in mjnemogym |
| `o200k_base.tiktoken` | `/root/tiktoken_cache/o200k_base.tiktoken` | vLLM 0.12 uses OpenAI tiktoken encoding |

### A10. "DO NOT TOUCH" — Base Image Protected Packages

Every `pip install` in the Dockerfile uses `--no-deps` to protect these.
If ANY of these get modified, all CUDA extensions (vLLM, flash_attn, TE, DeepEP)
will break because they depend on the exact ABI.

| Package | 26.02 Version | Why Protected |
|---------|--------------|---------------|
| `torch` | 2.10.0a0+b558c986e8.nv25.11 | NVIDIA custom build; ABI for all CUDA extensions |
| `torchvision` | 0.25.0a0+7a13ad0f | Must match torch |
| `pytorch-triton` | 3.5.0+gitde3506d2 | Must match torch (renamed from `triton` in 25.11.01) |
| `flash_attn` | 2.7.4.post1+25.11 | NVIDIA custom build for this CUDA/torch |
| `flashinfer-python` | 0.5.3 | Compiled for this CUDA/torch |
| `transformer_engine` | 2.12.0+5671fd36 | NVIDIA custom; megatron-core fused attention depends on it |
| `deep_ep` | 1.2.1+eb9cee7 | DeepEP CUDA extension for expert parallelism |
| `nvidia-modelopt` | 0.37.0 | NVIDIA model optimizer |
| `nvidia-resiliency-ext` | 0.4.1+cuda13 | NVIDIA resiliency |
| `nvfuser` | 0.2.34+gitfce1aec | Must match torch |
| `torchao` | 0.14.0+git | Must match torch |
| `torch_tensorrt` | 2.10.0a0 | Must match torch |
| `tensorrt` | 10.14.1.48 | System-level binary |
| `numpy` | 1.26.4 | C ABI dependency for many packages |
| `cuda-bindings` | 13.1.1 | System-level CUDA 13.0 |
| `cuda-python` | 13.1.1 | System-level CUDA 13.0 |
| `cupy-cuda12x` | 14.0.1 | Compiled for CUDA 13.0 |
| `apex` | 0.1 | NVIDIA custom mixed precision |
| `compressed-tensors` | 0.13.0 | Model loading dependency |

Non-pip but equally protected (from `/opt/` source paths, importable but not in pip):
| Package | Location | Version |
|---------|----------|---------|
| `megatron-core` | `/opt/Megatron-Bridge/3rdparty/Megatron-LM/` | 0.16.0 |
| `nemo-toolkit` | `/opt/NeMo/` | 2.7.0 |
| `megatron-bridge` | `/opt/Megatron-Bridge/` | (not in pip) |
| `transformer_engine` | `/opt/venv/lib/python3.12/site-packages/` | 2.12.0 |

---

## Part B: Personal Workflow Dependencies

Everything beyond the verl call stack: dev tools, shell config, monitoring,
data management, and operational requirements.

### B1. System Packages (apt)

| Package | Why |
|---------|-----|
| `pdsh` | Parallel shell commands across nodes (`p-run` alias) |
| `tmux` | Terminal multiplexer for long training sessions |
| `htop` | System monitoring |
| `vim` | Editor |

### B2. Shell Configuration

| File | Destination | Contents |
|------|------------|----------|
| `to_append.sh` | Appended to `/root/.bashrc` | `tnew`/`tat`/`tkl` tmux aliases, `gd` alias to cd to verl |
| `.tmux.conf` | `/root/.tmux.conf` | tmux config |

### B3. wandb Login

`wandb login df3cecbfc0874c8a352c40820becf4a15575614e` baked into image.
Training runs in `WANDB_MODE=offline` (no internet needed at runtime).
Logs written to `/root/myCodeLab/host/verl/wandb_my_dirs/`.

### B4. dist-packages Convenience Symlink

```
ln -s /usr/local/lib/python3.12/dist-packages /root/myCodeLab/dist-packages
```
Quick access to inspect installed packages.

### B5. Data Downloads (One-Time Host Setup)

Not in image. Done once on shared filesystem, accessed via mount at runtime.

| Item | Host Path | Size | How Downloaded |
|------|-----------|------|---------------|
| Moonlight-16B model | `downloads/models/Moonlight16B/` | 30GB (27 safetensors) | `huggingface-cli download moonshotai/Moonlight-16B-A3B` |
| DAPO-Math-17k dataset | `downloads/datasets/dapo_data/dapo-math-17k.parquet` | 286MB | `huggingface-cli download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17k` |
| AIME-2024 dataset | `downloads/datasets/dapo_data/aime-2024.parquet` | 29KB | `huggingface-cli download --repo-type dataset BytedTsinghua-SIA/AIME-2024` |

**Critical**: After downloading Moonlight-16B, edit `config.json`:
1. Remove `quantization_config` block
2. Set `num_nextn_predict_layers: 0` (MTP not supported)

### B6. `/etc/hosts` on Both Nodes

```
10.12.11.5 b31
10.12.11.6 b32
```

### B7. Proxy Configuration

Proxy at `http://jdtcom:709a64b73eb3@10.119.176.202:3128`.

| Function | Command | When Used |
|----------|---------|-----------|
| Shell proxy ON | `von` | pip/curl/huggingface downloads |
| Shell proxy OFF | `voff` | Normal operation |
| Docker daemon proxy ON | `dvon` | `docker build` (pip needs internet) |
| Docker daemon proxy OFF | `dvoff` | `docker push`/`pull` to Aliyun (registry unreachable via proxy) |

**Critical constraint**: `dvon` restarts docker daemon. Running containers survive
but cannot `docker compose up` during restart. Also, Aliyun registry
(`registry.cn-hangzhou.aliyuncs.com`) is NOT reachable through the proxy.
Build/push workflow: `dvon` → build → `dvoff` → push.

### B8. Editable Install Runtime Behavior

Both `verl` and `mjnemogym` are installed as editable (`pip install -e .`).

**How it works at runtime:**
1. During `docker build`, `COPY verl /root/myCodeLab/host/verl/` puts source in image
2. `pip install --no-deps -e .` creates `.pth` files in `dist-packages/` recording the path `/root/myCodeLab/host/verl`
3. At runtime, docker-compose mounts the host project dir over `/root/myCodeLab/host/`
4. The mount overlays the COPY'd content, but the `.pth` files (in `dist-packages/`, NOT under the mount) still point to `/root/myCodeLab/host/verl` — which now resolves to the live mounted source
5. Result: code changes on host are immediately visible inside container. No re-install needed

**Critical constraint**: The COPY destination names in the Dockerfile MUST match
the directory names in the mounted host directory. Otherwise the `.pth` paths
break:
- `COPY verl /root/myCodeLab/host/verl/` ↔ host has `verl/` ✓
- `COPY MJ_NEMO_GYM /root/myCodeLab/host/MJ_NEMO_GYM/` ↔ host has `MJ_NEMO_GYM/` ✓

If the Dockerfile used a different name (e.g., `mjnemogym/`), the `.pth` would
record `/root/myCodeLab/host/mjnemogym/` but the mount would provide
`MJ_NEMO_GYM/` → import failure (this was Bug #4's root cause in v1).

### B9. Runtime Path Map (Complete)

Docker-compose volumes:
```yaml
volumes:
  - /mnt/public:/public
  - /mnt/public/lichang93/st_verl_dockerfile:/root/myCodeLab/host
```

| Container Path | Resolves To (Host) | Access |
|---------------|-------------------|--------|
| `/root/myCodeLab/host/verl/` | `st_verl_dockerfile/verl/` | Read (editable source) |
| `/root/myCodeLab/host/MJ_NEMO_GYM/` | `st_verl_dockerfile/MJ_NEMO_GYM/` | Read (editable source) |
| `/root/myCodeLab/host/downloads/` | symlink → `stCodeLab/downloads/` | Read (model + datasets) |
| `/root/myCodeLab/host/verl/wandb_my_dirs/` | `st_verl_dockerfile/verl/wandb_my_dirs/` | Write (wandb logs) |
| `/root/myCodeLab/host/verl/ckpts/` | `st_verl_dockerfile/verl/ckpts/` | Write (checkpoints) |
| `/root/myCodeLab/host/verl/my_scripts/logs/` | `st_verl_dockerfile/verl/my_scripts/logs/` | Write (training logs) |
| `/opt/vllm/` | Image layer (not mounted) | Read (vLLM editable install) |
| `/usr/local/share/nltk_data/` | Image layer (COPY'd) | Read |
| `/root/tiktoken_cache/` | Image layer (COPY'd) | Read |

### B10. Lessons Learned (All Bugs)

Complete list of every issue encountered during development:

| # | Phase | Summary | Root Cause | Resolution |
|---|-------|---------|------------|------------|
| B-1 | Build | Tsinghua pip mirror 503 | Docker proxy blocks tuna mirror | Remove `pip config set global.index-url` for tuna; use default pypi.org |
| B-2 | Build | `grouped_gemm` CUDA version mismatch | pip build isolation downloads its own torch (CUDA 12.6) | `--no-build-isolation` for all CUDA extensions |
| B-3 | Build | vLLM version detection failure | `.git` stripped from source | `SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0` |
| B-4 | Build | Docker 127-layer limit exceeded | Base has 107 layers + 21 new = 128 | Consolidated to 14 new layers (total 121) |
| B-5 | Build | 3.8GB build context | Compiled `.so` files and `.deps/` in vLLM source | Clean before COPY (reduced to 26MB) |
| B-6 | Build | `--provenance` flag error | Legacy builder (`DOCKER_BUILDKIT=0`) doesn't support it | Remove flag |
| 1 | Runtime | `/root/myCodeLab/host` symlink broken | Dockerfile hardcoded `ln -s /public/lichang93/stCodeLab` | v2: plain dir + compose mount |
| 2 | Runtime | Dataset download 401 | Missing `--repo-type dataset` in huggingface-cli | Add flag |
| 3 | Runtime | `FileExistsError: /root/myCodeLab/host` | `mkdir -p` on symlink to non-existent target | v2: plain dir |
| 4 | Runtime | `ModuleNotFoundError: verl` | Editable install path mismatch after mount | v2: COPY dest names match host dir names |
| 5 | Runtime | `ModuleNotFoundError: pyvers` | `tensordict` dep, `--no-deps` skipped it | Add to Dockerfile |
| 6 | Runtime | `ModuleNotFoundError: math_verify` | `mjnemogym` dep | Add to Dockerfile |
| 7 | Runtime | `ModuleNotFoundError: latex2sympy2_extended` | `math_verify` dep | Add to Dockerfile |
| 8 | Runtime | `ModuleNotFoundError: openapi_schema_validator` | `mjnemogym` dep | Add to Dockerfile |
| 9 | Runtime | `ModuleNotFoundError: orjson` | `verl.utils.tracking` dep | Add to Dockerfile |
| 10 | Runtime | 27 layers not divisible by PP=2 | Moonlight-16B architecture | `train_pp=1` |
| 11 | Runtime | VPP requires PP>1 | VPP incompatible with PP=1 | `vpp=null` |
| 12 | Runtime | NCCL wrong IP (bond0.200 VLAN) | Missing `/dev/infiniband/`, NCCL falls back to Socket | `privileged: true` |
| 13 | Runtime | YAML folded scalar broke ray start | `>` splits at indentation boundaries | List format `["-c", ">- ..."]` |
| 14 | Runtime | NCCL IB `status=12 vendor err 129` | GID index 0 = link-local fe80:: (not routable) | `NCCL_IB_GID_INDEX=3` (RoCEv2) |
| 15 | Runtime | vLLM `No common block size for 16` | `is_device_capability(100)` exact match fails for sm_103 | `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` |
| 16 | Image | Different image IDs on b31 vs b32 | Local build vs tar load | Rebuild + registry push/pull |
| 17 | Image | Fragile stCodeLab symlinks | Image assumed host path structure | v2: no symlinks in image |
