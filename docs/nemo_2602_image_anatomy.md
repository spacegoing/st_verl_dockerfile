# Anatomy of nvcr.io/nvidia/nemo:26.02 — Layer-by-Layer Analysis

**Date**: 2026-03-13
**Source**: `docker history nvcr.io/nvidia/nemo:26.02 --no-trunc` (303 layers)

---

## 1. How uv Is Used in the Official Image

### Two Modes of uv

uv has two distinct usage modes. NVIDIA uses **both** in this image:

| Mode | Command | What it does | Config source |
|------|---------|-------------|---------------|
| **Project mode** | `uv sync` | Reads `pyproject.toml` + `uv.lock`, resolves all deps, installs to target venv | `pyproject.toml` in project dir |
| **pip-compat mode** | `uv pip install` | Installs individual packages to active venv, no lockfile | CLI args only |

### uv Project Mode (How NVIDIA Uses It)

NVIDIA uses uv's project mode for the NeMo ecosystem. The project is at `/opt/NeMo-FW/`:

```
/opt/NeMo-FW/
  pyproject.toml    # Declares all NeMo dependencies + groups
  uv.lock           # Pinned versions for reproducible builds
  _version.py       # Version info
```

The key ENV that decouples the venv from the project directory:
```bash
UV_PROJECT_ENVIRONMENT=/opt/venv   # Without this, uv sync would install to /opt/NeMo-FW/.venv/
```

Build commands:
```bash
# Step 1: Megatron-Bridge project (has its own pyproject.toml + uv.lock)
cd /opt/Megatron-Bridge && uv sync --all-extras --all-groups

# Step 2: NeMo-FW project (the main one)
cd /opt/NeMo-FW && uv sync --no-cache-dir --all-groups --inexact
```

The `--inexact` flag is critical: it means "add these packages but don't remove packages
that aren't in the lockfile." This preserves everything pip installed earlier (torch, CUDA
packages, etc.) in both `/usr/local/` and `/opt/venv/`.

### uv pip-compat Mode (What We Use)

`uv pip install` is simpler — it's pip-compatible syntax that installs to the active venv:
```bash
# Respects VIRTUAL_ENV=/opt/venv → installs to /opt/venv/lib/python3.12/site-packages/
uv pip install --no-deps transformers==4.57.3
```

No `pyproject.toml` or `uv.lock` needed. This is equivalent to `pip install` but targets
the correct location.

### Why Runtime Auto-Resolves to /opt/venv/ (Root Cause)

Three ENV directives baked into the Docker image — no shell activation needed:

```dockerfile
# Set in NeMo Dockerfile layers:
ENV VIRTUAL_ENV=/opt/venv                    # Tells Python/uv which venv is active
ENV PATH=/opt/venv/bin:...:$PATH             # Makes python3 → /opt/venv/bin/python3
ENV UV_PROJECT_ENVIRONMENT=/opt/venv         # Tells uv sync where to install
```

At container startup:
```
1. Shell sees PATH → /opt/venv/bin is FIRST
   → `python3` = /opt/venv/bin/python3

2. /opt/venv/bin/python3 is a symlink → /usr/bin/python3.12 (SAME binary)
   But Python detects it's running from a venv directory

3. Python reads /opt/venv/pyvenv.cfg:
   include-system-site-packages = true

4. Python builds sys.path:
   /opt/venv/lib/python3.12/site-packages/    ← searched FIRST (venv wins)
   /usr/local/lib/python3.12/dist-packages/   ← searched SECOND (system loses)
```

### Why `pip install` Goes to the Wrong Place

Despite the venv being "active", the system pip binary has a hardcoded shebang:
```bash
$ which pip
/usr/local/bin/pip

$ head -1 /usr/local/bin/pip
#!/usr/bin/python           # Bypasses venv, runs as system python directly
```

There is no `/opt/venv/bin/pip`. The image's pip is the **system pip** and always writes to
`/usr/local/lib/python3.12/dist-packages/` regardless of VIRTUAL_ENV.

In contrast, `uv pip install` reads `VIRTUAL_ENV=/opt/venv` and writes there:
```
$ uv pip install --dry-run --no-deps requests
Using Python 3.12.3 environment at: /opt/venv    ← correct!
```

---

## 2. Image Build Phases (Bottom-Up Chronological)

The 303 layers build in 6 distinct phases. Reading `docker history` top-down shows newest
first; the actual build order is bottom-up (L303→L1).

### Phase A: Ubuntu + RDMA/Networking Infrastructure (L303→L251)

**What**: Ubuntu 24.04 + networking stack for multi-node GPU communication.

| Component | Version | What It Does |
|-----------|---------|-------------|
| rdma-core | 56.0 | RDMA verbs library (InfiniBand/RoCE kernel interface) |
| GDRCopy | 2.5.1 | GPU Direct RDMA — CPU↔GPU memory copies via PCIe BAR1 |
| HPC-X | 2.24.1 | NVIDIA's HPC stack bundle containing UCX + OpenMPI |
| UCX | 1.19.0 | Unified Communication X — transport layer for RDMA/RoCE |
| OpenMPI | 4.1.7 | MPI implementation (used by NCCL, torch.distributed) |
| EFA | 1.43.1 | Elastic Fabric Adapter (AWS networking, also works on-prem) |
| AWS-OFI-NCCL | 1.17.0 | NCCL over EFA/libfabric plugin |
| DOCA | 3.1.0 | NVIDIA networking SDK (BlueField DPU, ConnectX smart NICs) |

### Phase B: CUDA Toolkit + Libraries (L250→L183)

**What**: Full CUDA development toolkit.

| Component | Version | What It Does |
|-----------|---------|-------------|
| CUDA | 13.0.2 | Compiler + runtime (nvcc, cudart) |
| cuDNN | 9.15.0 | Deep learning convolution/attention primitives |
| NCCL | 2.28.8 | Multi-GPU/multi-node collective communication |
| NVSHMEM | 3.4.5 | GPU-initiated RDMA (system libs, not Python bindings) |
| cuBLAS | 13.1.0 | GPU linear algebra |
| cuSPARSE | 12.6.3 | Sparse matrix ops |
| cuSPARSELt | 0.8.1 | Structured sparsity (2:4 pruning acceleration) |
| TensorRT | 10.14.1 | Inference optimization/compilation |
| NSight Systems | 2025.5.1 | Profiling |
| NSight Compute | 2025.3.1 | Kernel-level profiling |

### Phase C: PyTorch Ecosystem (L182→L91)

**What**: PyTorch and core ML libraries — pure pip installs, no venv yet.

| Component | Version | Install Method | What It Does |
|-----------|---------|---------------|-------------|
| **PyTorch** | **2.10.0a0+nv25.11** | `pip install torch*.whl` | Core framework |
| **apex** | **0.1** | `pip install apex*.whl` | Mixed-precision training |
| **torch_tensorrt** | **2.10.0a0** | `pip install torch_tensorrt*.whl` | TensorRT integration |
| **flash_attn** | **2.7.4.post1** | `pip install flash_attn*.whl` | Flash Attention kernels |
| **TransformerEngine** | **2.9** | `pip install --no-build-isolation` (compiled from source) | FP8 training, fused attention |
| **nvidia-modelopt** | **0.37.0** | `pip install` from NVIDIA PyPI | Quantization/pruning toolkit |
| **nvidia-resiliency-ext** | **0.4.1+cuda13** | `pip install` from NVIDIA GitLab | Fault tolerance for training |
| nvfuser | 0.2.34 | `pip install --no-build-isolation` (compiled) | JIT fusion compiler |
| torchao | 0.14.0 | `pip install --no-build-isolation` (compiled) | PyTorch Architecture Optimization |
| lightning-thunder | - | `pip install --no-build-isolation` | Thunder compiler |
| cuBLASMp | - | Build script | Multi-process cuBLAS |
| DALI | 1.52.0 | `pip install` with NVIDIA index | Data loading pipeline |
| pytorch-triton | 3.5.0 | (bundled with torch wheel) | Triton compiler for PyTorch |

**All installs go to `/usr/local/lib/python3.12/dist-packages/`** — no venv exists yet.

Also installs: numpy, scipy, protobuf, pybind11, Cython, mkl, jupyterlab, tensorboard.

### Phase D: NeMo Framework Base Setup (L90→L62)

**What**: Transition point — sets up uv venv and NeMo framework infrastructure.

```dockerfile
# L82: Set product name
ENV NVIDIA_PRODUCT_NAME=NeMo Framework

# L66-65: Configure uv target
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache

# L64: Add /opt/venv/bin to PATH (FIRST time)
ENV PATH=/opt/venv/bin:...

# L63: uv link mode
ENV UV_LINK_MODE=copy

# L62: Install uv 0.8.22 + CREATE the venv
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | XDG_BIN_HOME=/usr/local/bin sh
uv venv /opt/venv --system-site-packages
```

**Key**: `--system-site-packages` makes the venv see everything in `/usr/local/` too.
At this point `/opt/venv/lib/python3.12/site-packages/` is essentially EMPTY — it will
be populated by later uv sync steps.

### Phase E: DeepEP + vLLM + TensorRT-LLM (L61→L37) — The Critical Infra Layer

**What**: Performance-critical packages that need CUDA compilation + RDMA.

#### cuDNN Upgrade (L54)
```bash
# Upgrades cuDNN from 9.15 (Phase B) to 9.18
REINSTALL_CUDNN=True CUDNN_VERSION=9.18.0.50+cuda13.1
INTERNAL=1 DEVEL=1 /nvidia/build-scripts/installCUDNN.sh
```
**Why**: cuDNN 9.18 adds MLA (Multi-head Latent Attention) fused kernels for sm>=100.
Required for DeepSeek-V3/Moonlight attention patterns on B300.

#### DeepEP (L49)
```bash
# 1. Rebuild rdma-core v60.0 from source (image ships v56.0)
git clone https://github.com/linux-rdma/rdma-core.git
cd rdma-core && git checkout tags/v60.0 && sh build.sh

# 2. Install NVSHMEM Python bindings
pip install --no-cache-dir nvidia-nvshmem-cu13==3.4.5

# 3. Build DeepEP from source with GPU Direct RDMA support
cd DeepEP && pip install --no-cache-dir --no-build-isolation -v .
```
Compiled with `TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0"`.

ENV vars set:
```bash
RDMA_CORE_HOME=/opt/rdma-core/build    # Points to rebuilt v60.0
HYBRID_EP_MULTINODE=1                    # Enable multi-node expert parallelism
```

#### vLLM + xgrammar + TensorRT-LLM (L47)
```bash
pip install --no-deps xgrammar==0.1.25
pip install /tmp/vllm/vllm*.whl          # vLLM 0.14.2 (pre-built wheel)
pip install /tmp/tensorrt_llm/tensorrt_llm*.whl
```
These go to `/usr/local/` (system pip, no venv pip).

### Phase F: Megatron + NeMo Ecosystem via uv (L36→L1)

**What**: The NeMo-specific Python ecosystem. This is the "NeMo stuff."

#### Second uv Install + Venv Recreation (L37)
```bash
# Reinstall uv at newer version, recreate venv
curl -LsSf https://astral.sh/uv/0.7.2/install.sh | sh    # → ~/.local/bin/uv
uv venv /opt/venv --system-site-packages                   # Recreates venv
```
Previous venv (from L62) is wiped and recreated. But system site-packages are preserved
because they're in `/usr/local/`.

#### Megatron-Bridge + Megatron-Core (L36→L29)
```bash
cd /opt/Megatron-Bridge
uv sync --only-group build                    # Build tools first
uv sync --link-mode copy --all-extras --all-groups  # Full install
```
This installs megatron-core 0.16.0 and megatron-bridge into `/opt/venv/`.

#### NeMo + Ecosystem Git Clones (L28→L15)
```bash
git clone NeMo, Export-Deploy, Evaluator, Run    # To /opt/
git checkout specific commits for each
```

#### NeMo-FW Full Sync (L13)
```bash
cd /opt/NeMo-FW
uv sync --no-cache-dir --all-groups --inexact
# Also patches NLTK
```
**This is the big one** — installs 900+ Python packages into `/opt/venv/` based on
NeMo-FW's `pyproject.toml` + `uv.lock`. This is where packages like transformers 4.57.6,
wandb 0.24.0, pydantic 2.13.0b1, flashinfer 0.6.4, etc. get installed.

The `--inexact` flag preserves all system pip packages (torch, flash_attn, etc.).

#### Security Patches (L12)
```bash
pip install nvidia-cudnn-frontend==1.15.0 \
    starlette==0.49.1 urllib3>=2.6.0 aiohttp>=3.13.3 \
    setuptools>=81.0.0 protobuf>=6.33.5 pillow==12.1.1
```
These are **pip install** (not uv) → go to `/usr/local/` → most get shadowed by
venv versions installed in L13. This is the "NVIDIA oversight" we identified earlier.

#### Final ENV (L1)
```bash
ENV NVTE_CPU_OFFLOAD_V1=1    # TransformerEngine CPU offloading
```

---

## 3. What Each Phase Provides (Summary)

| Phase | Layers | Size Estimate | Contains | Needed for verl+DeepEP? |
|-------|--------|--------------|----------|------------------------|
| A: Ubuntu + RDMA | L303→L251 | ~2GB | rdma-core, GDRCopy, HPC-X, UCX, OpenMPI, EFA | **YES** (RDMA/RoCE) |
| B: CUDA Toolkit | L250→L183 | ~8GB | CUDA 13.0, cuDNN 9.15, NCCL, NVSHMEM system libs | **YES** |
| C: PyTorch | L182→L91 | ~6GB | torch, apex, flash_attn, TransformerEngine, modelopt | **YES** (core training) |
| D: NeMo Base | L90→L62 | ~0.1GB | uv setup, venv creation, PATH/ENV config | Needed for uv infra only |
| E: DeepEP+vLLM | L61→L37 | ~3GB | cuDNN→9.18, DeepEP+rdma-core v60, vLLM, TRT-LLM | **YES** (critical) |
| F: NeMo Ecosystem | L36→L1 | ~4GB | megatron-core, NeMo, 900+ packages via uv sync | **PARTIAL** (need megatron-core) |

### What NeMo Adds Beyond PyTorch (Performance-Relevant)

| Addition | Phase | Performance Impact |
|----------|-------|-------------------|
| cuDNN 9.15→9.18 upgrade | E | **Critical**: MLA fused attention on sm>=100 |
| DeepEP + rdma-core v60.0 | E | **Critical**: MoE expert parallelism via GPU-initiated RDMA |
| nvidia-nvshmem-cu13 pip bindings | E | **Critical**: Python interface to NVSHMEM for DeepEP |
| TransformerEngine 2.9→2.12 (via uv sync) | F | **Significant**: Newer TE with FP8 improvements |
| flashinfer 0.6.4 (via uv sync) | F | **Significant**: Newer attention kernels |
| megatron-core 0.16.0 | F | **Required**: MoE, pipeline parallelism |
| NeMo 2.7.0 | F | Not needed for verl |
| NeMo-Run, Evaluator, Export-Deploy | F | Not needed |
| 800+ other Python packages | F | Not needed |

---

## 4. Alternative Base Image Consideration

### Option: Use `nvcr.io/nvidia/pytorch:25.11-py3` Instead

The PyTorch image contains Phases A+B+C (Ubuntu, CUDA, PyTorch). You'd need to add:

**Must rebuild from NeMo's Phase E:**
1. cuDNN upgrade to 9.18 (critical for MLA attention)
2. rdma-core v60.0 from source (DeepEP needs newer verbs)
3. nvidia-nvshmem-cu13==3.4.5 (Python NVSHMEM bindings)
4. DeepEP compilation with `--no-build-isolation`
5. Set `RDMA_CORE_HOME`, `HYBRID_EP_MULTINODE=1`

**Must install from NeMo's Phase F:**
6. megatron-core 0.16.0 (via Megatron-Bridge uv sync or manual pip install)

**Benefits of PyTorch base:**
- No uv/pip dual-layer confusion (just pip)
- ~4GB smaller image (no NeMo ecosystem)
- Simpler dependency management
- No 900+ unnecessary packages

**Risks of PyTorch base:**
- Must exactly replicate DeepEP build steps (rdma-core version, NVSHMEM version, patches)
- NVIDIA patches DeepEP with `/opt/deepep.patch` — we'd need that patch
- cuDNN upgrade is non-trivial (NVIDIA uses internal build scripts)
- TransformerEngine may need recompilation if version 2.9 isn't sufficient
- Megatron-core may have hidden dependencies resolved by NeMo-FW's lockfile
- Less tested configuration — NeMo image is NVIDIA's validated stack

### Verdict

The NeMo image is heavy but provides a **validated, working DeepEP+NVSHMEM+RDMA stack**.
Rebuilding this from the PyTorch base is doable but requires careful replication of Phase E
(~50 lines of complex build commands with specific patches).

If image size and dependency cleanliness matter more than build simplicity, PyTorch base
is the right call. If validated infrastructure matters more, stick with NeMo.
