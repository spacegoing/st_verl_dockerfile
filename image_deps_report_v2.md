# Exhaustive 3-Image Dependency Report (v2)

**Date**: 2026-03-12
**Images**:
- **A** = `nvcr.io/nvidia/nemo:25.11.01` (official base, old)
- **B** = `nvcr.io/nvidia/nemo:26.02` (official base, new)
- **C** = `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` (built 2026-03-12 06:31 UTC, SHA: 6a06782cdad6)

**Image C Build Evidence**: Docker history shows all 15 layers match `Dockerfile.ncr.26.02.mydev` exactly, including L5 with `yappi itsdangerous gprof2dot pydot` added on 2026-03-12.

---

## 1. System-Level Components

| Component | A (25.11.01) | B (26.02) | C (myverl) | Notes |
|-----------|-------------|-----------|------------|-------|
| **Python** | 3.12.3 | 3.12.3 | 3.12.3 | Same |
| **CUDA (nvcc)** | 13.0.88 | 13.0.88 | 13.0.88 | Same |
| **cuDNN** | 9.13.1 | **9.18.0** | **9.18.0** | B=C upgraded (needed for B300 MLA) |
| **NCCL** | 2.28.3 | **2.28.9** | **2.28.9** | B=C upgraded |
| **GCC** | 13.3.0 | 13.3.0 | 13.3.0 | Same |
| **CMake** | 3.31.6 | 3.31.6 | 3.31.6 | Same |
| **Ubuntu** | 24.04.3 | 24.04.3 | 24.04.3 | Same |
| **NVIDIA Build** | 243847214 | **271855353** | **271855353** | B=C (26.02 build) |

---

## 2. Core ML Frameworks

| Package | A (25.11.01) | B (26.02) | C (myverl) | Notes |
|---------|-------------|-----------|------------|-------|
| **PyTorch** | 2.9.0a0+nv25.09 | **2.10.0a0+nv25.11** | **2.10.0a0+nv25.11** | B=C upgraded |
| **torchvision** | 0.24.0a0 | **0.25.0a0** | **0.25.0a0** | B=C upgraded |
| **pytorch-triton** | 3.4.0 (`triton` pkg) | **3.5.0** (`pytorch-triton`) | **3.5.0** | B=C, pkg renamed |
| **flash_attn** | 2.7.4.post1 | 2.7.4.post1 | 2.7.4.post1 | Same version, different build |
| **flashinfer** | 0.5.3 (pip) | 0.5.3 (pip) | 0.5.3 (pip) | pip says 0.5.3, runtime reports 0.6.4 for B/C |
| **transformer_engine** | 2.9.0+70f53666 | **2.12.0+5671fd36** | **2.12.0+5671fd36** | B=C upgraded |
| **nvfuser** | (in /opt/pytorch) | 0.2.34+gitfce1aec | 0.2.34+gitfce1aec | B=C added to pip |
| **torchao** | 0.13.0+git | **0.14.0+git** | **0.14.0+git** | B=C upgraded |
| **torch_tensorrt** | 2.9.0a0 | **2.10.0a0** | **2.10.0a0** | B=C upgraded |

---

## 3. vLLM (CRITICAL DIFFERENCE)

| | A (25.11.01) | B (26.02) | C (myverl) |
|--|-------------|-----------|------------|
| **Version** | 0.10.1.1+nv25.09 | **0.14.2.dev0** | **0.12.0+cu130** |
| **Install type** | dist-packages | dist-packages | **editable @ /opt/vllm** |
| **Source dir** | `/opt/vllm/` (exists) | **NO /opt/vllm/** | `/opt/vllm/` (COPY'd) |

**Key**: C uses custom vLLM 0.12.0 (editable) while B uses 0.14.2. The Dockerfile uninstalls B's vLLM, copies 0.12.0 source, and builds it.

---

## 4. NeMo Ecosystem

| Package | A (25.11.01) | B (26.02) | C (myverl) |
|---------|-------------|-----------|------------|
| **megatron-core** | 0.16.0rc0 (pip, editable @ `/opt/megatron-lm`) | **0.16.0** (importable, NOT in pip, @ `/opt/Megatron-Bridge/3rdparty/Megatron-LM/`) | **0.16.0** (B=C) |
| **nemo-toolkit** | 2.6.0rc0 (pip, editable @ `/opt/NeMo`) | **2.7.0** (importable, NOT in pip, @ `/opt/NeMo`) | **2.7.0** (B=C) |
| **NeMo-FW** | 25.11rc3 (pip, editable @ `/opt/NeMo-FW`) | present but **NOT in pip** | present but **NOT in pip** (B=C) |
| **megatron-bridge** | 0.2.0 (pip, editable @ `/opt/Megatron-Bridge`) | **NOT in pip** (@ `/opt/Megatron-Bridge`) | **NOT in pip** (B=C) |
| **nvidia-modelopt** | 0.37.0 (editable @ `/opt/TensorRT-Model-Optimizer`) | 0.37.0 (dist-packages) | 0.37.0 (B=C) |
| **nvidia-resiliency-ext** | 0.4.1 (editable @ `/opt/nvidia-resiliency-ext`) | 0.4.1+cuda13 (dist-packages) | 0.4.1+cuda13 (B=C) |

**Key**: In B/C, NeMo ecosystem packages are **NOT registered in pip metadata** but ARE importable via source paths. `megatron-core` moved from `/opt/megatron-lm/` (A) to `/opt/Megatron-Bridge/3rdparty/Megatron-LM/` (B/C).

---

## 5. `/opt/` Directory Structure Changes

| Directory | A (25.11.01) | B (26.02) | C (myverl) |
|-----------|-------------|-----------|------------|
| `/opt/vllm/` | source (vllm-src, vllm-flash-attn) | **REMOVED** | **RESTORED** (COPY'd by Dockerfile) |
| `/opt/megatron-lm/` | megatron-core source | **REMOVED** (now inside Megatron-Bridge) | **REMOVED** (B=C) |
| `/opt/xformers/` | xformers source | **REMOVED** | **REMOVED** (B=C) |
| `/opt/nvidia-resiliency-ext/` | editable source | **REMOVED** | **REMOVED** (B=C) |
| `/opt/TensorRT-Model-Optimizer/` | modelopt source | **REMOVED** | **REMOVED** (B=C) |
| `/opt/rdma-core/` | not present | **NEW** | **NEW** (B=C) |

---

## 6. Package Count Summary

- **A (25.11.01)**: 559 packages
- **B (26.02)**: 349 packages (210 fewer than A)
- **C (myverl)**: 373 packages (24 more than B, 186 fewer than A)

**Why B has fewer packages**: NVIDIA moved to a "leaner base" philosophy in 26.02, removing many dev/optional packages. C adds back 24 packages needed for verl/Gym.

---

## 7. Key Version Changes: A→B that C Inherits (B==C, A!=B)

| Package | A (25.11.01) | B=C (26.02/myverl) | Impact |
|---------|-------------|-------------------|--------|
| **ray** | 2.52.1 | **2.54.0** | Minor upgrade |
| **protobuf** | 4.25.8 | **6.33.5** | MAJOR: 4.x→6.x breaking change |
| **transformers** | 4.56.0 | **4.57.3** | C explicitly pins 4.57.3 in Dockerfile L5 |
| **openai** | 2.11.0 | **2.21.0** | Major upgrade |
| **peft** | 0.13.2 | **0.18.1** | Major upgrade |
| **compressed-tensors** | 0.10.2 | **0.13.0** | Upgrade |
| **tensorrt** | 10.13.3.9 | **10.14.1.48** | Upgrade |
| **setuptools** | 79.0.1 | **81.0.0** | Upgrade |
| **aiohttp** | 3.13.2 | **3.13.3** | Minor |
| **fastapi** | 0.124.2 | **0.121.3** | Downgrade (!) |
| **grpcio** | 1.76.0 | **1.78.1** | Upgrade |
| **httpx** | 0.27.2 | **0.28.1** | Upgrade |
| **huggingface-hub** | 0.36.0 | **0.36.2** | Minor |
| **ipykernel** | 6.30.1 | **7.1.0** | Major |
| **ipython** | 9.8.0 | **9.7.0** | Downgrade (!) |
| **matplotlib** | 3.10.8 | **3.10.7** | Downgrade (!) |

---

## 8. Packages ONLY in C (myverl) — Dockerfile Additions

These 24 packages are added by `Dockerfile.ncr.26.02.mydev`:

**L5 pip installs**:
- `wandb` (0.25.0) — was in A, removed from B, re-added by C
- `gpustat` (1.1.1)
- `codetiming` (1.4.0)
- `tensordict` (0.7.0)
- `mathruler` (0.1.0)
- `pylatexenc` (2.10)
- `torchdata` (0.11.0a0)
- `hydra-core` (1.3.2) — was in A, removed from B, re-added by C
- `bitsandbytes` (0.49.2) — was in A, removed from B, re-added by C
- `orjson` (3.10.17)
- `anthropic` (0.45.1)
- `model-hosting-container-standards` (0.1.0)
- `pyvers` (1.1.1)
- `math-verify` (0.1.0)
- `latex2sympy2-extended` (1.10.1)
- `openapi-schema-validator` (0.7.1)
- `langdetect` (1.0.9)
- `absl-py` (2.2.0)
- `immutabledict` (4.2.1)
- `yappi` (1.6.4) — **NEW in iter2**
- `itsdangerous` (2.2.0) — **NEW in iter2**
- `gprof2dot` (2024.6.6) — **NEW in iter2**
- `pydot` (3.0.4) — **NEW in iter2**

**L6 mbridge install**:
- `mbridge` (0.1.0, editable)

**L13 verl/Gym/verifiable-instructions installs**:
- `verl` (0.8.0.dev0, editable @ `/root/myCodeLab/host/verl`)
- `nemo-gym` (0.2.0rc0, editable @ `/root/myCodeLab/host/Gym`)
- `verifiable-instructions` (0.1.0)

---

## 9. Packages in A (25.11.01) ONLY — Removed from Both B and C

These 210 packages were in A but removed from B (and thus not in C):

**Notable removals**:
- `xformers` (0.0.32+nv25.09) — was editable in A, completely removed in B/C
- Many dev/build tools: `autoconf`, `automake`, `libtool`, `m4`, `pkg-config`
- Many Python dev packages: `build`, `installer`, `wheel`, `poetry-core`
- Jupyter extensions: `jupyter-lsp`, `jupyterlab-git`, `jupyterlab-nvdashboard`
- Monitoring: `nvitop`, `py3nvml`
- Testing: `pytest-timeout`, `pytest-rerunfailures`
- Docs: `sphinx`, `sphinx-rtd-theme`, `myst-parser`
- Misc: `boto3`, `s3transfer`, `docker`, `kubernetes`

(Full list of 210 packages omitted for brevity — see raw pip freeze diffs)

---

## 10. Critical Differences: C vs B (Where myverl Diverges)

| Package | B (26.02) | C (myverl) | Reason |
|---------|-----------|------------|--------|
| **vllm** | 0.14.2.dev0 | **0.12.0+cu130** | Dockerfile L1-L3: uninstall B's vLLM, build custom 0.12.0 |
| **transformers** | 4.56.0 | **4.57.3** | Dockerfile L5: explicit pin |

All other packages: **B == C** (C inherits from B).

---

## 11. Packages in B but NOT in C — Removed by Dockerfile

Only **1 package**: `vllm` (0.14.2.dev0) — explicitly uninstalled in Dockerfile L1.

---

## 12. Summary of Changes

### A → B (25.11.01 → 26.02)
- **PyTorch**: 2.9.0 → 2.10.0
- **cuDNN**: 9.13.1 → 9.18.0 (needed for B300 MLA)
- **NCCL**: 2.28.3 → 2.28.9
- **vLLM**: 0.10.1.1 → 0.14.2
- **transformer_engine**: 2.9.0 → 2.12.0
- **ray**: 2.52.1 → 2.54.0
- **protobuf**: 4.25.8 → 6.33.5 (BREAKING)
- **NeMo**: 2.6.0rc0 → 2.7.0
- **megatron-core**: 0.16.0rc0 → 0.16.0 (moved to Megatron-Bridge/3rdparty)
- **Removed**: 210 packages (xformers, wandb, hydra-core, bitsandbytes, dev tools, etc.)

### B → C (26.02 → myverl)
- **vLLM**: 0.14.2 → **0.12.0** (custom build, editable)
- **transformers**: 4.56.0 → **4.57.3** (explicit pin)
- **Added**: 24 packages (wandb, gpustat, verl, Gym, mbridge, yappi, itsdangerous, gprof2dot, pydot, etc.)

---

## 13. Compatibility Notes

### protobuf 4.x → 6.x
B/C use protobuf 6.33.5 (major version jump from A's 4.25.8). This can break serialization if you have pickled protobuf objects from A.

### xformers Removal
`xformers` (0.0.32+nv25.09) was in A but completely removed from B/C. If your code imports xformers, it will fail. The base image no longer includes it.

### vLLM Custom Build
C uses vLLM 0.12.0 (custom, editable) instead of B's 0.14.2. This is intentional for compatibility with verl's vLLM integration.

### NeMo Ecosystem Metadata
In B/C, `megatron-core`, `nemo-toolkit`, `NeMo-FW`, `megatron-bridge` are **NOT in pip metadata** but ARE importable. `pip show` will fail, but `import megatron.core` works.

---

## 14. Dockerfile Layer Verification

Image C build history (2026-03-12 06:31 UTC) matches `Dockerfile.ncr.26.02.mydev` exactly:

| Layer | Time | Size | Dockerfile Line |
|-------|------|------|-----------------|
| Base | 02-23 19:03 | — | `FROM nvcr.io/nvidia/nemo:26.02` |
| L1 | 03-11 13:00 | 631kB | `pip uninstall -y vllm && pip install setuptools_scm` |
| L2 | 03-12 05:42 | 27MB | `COPY vllm /opt/vllm` |
| L3 | 03-12 05:42→06:10 | 3.95GB | vLLM 0.12.0 build (~27min) |
| L4 | 03-12 06:10→06:22 | 622MB | `grouped_gemm`, `causal_conv1d`, `mamba_ssm` |
| L5 | 03-12 06:22→06:23 | 444MB | apt + pip (includes `yappi itsdangerous gprof2dot pydot`) |
| L6 | 03-12 06:23→06:24 | 1.69MB | `COPY mbridge` + install |
| L7-L9 | 03-12 06:24→06:27 | 19.8MB | Config files, ENV, workspace setup |
| L10-L12 | 03-12 06:27→06:29 | 114.4MB | `COPY verl`, `COPY Gym`, `COPY verifiable-instructions` |
| L13 | 03-12 06:29→06:30 | 803kB | Install verl, Gym, verifiable-instructions (editable) |
| L14-L15 | 03-12 06:30→06:31 | 0B | `WORKDIR`, `CMD` |

**Confirmation**: L5 includes `yappi itsdangerous gprof2dot pydot` — proving this is the latest Dockerfile build with iter2 Gym server dependencies.

---

**End of Report**
