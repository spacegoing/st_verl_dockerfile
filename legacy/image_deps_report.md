# Exhaustive 3-Image Dependency Investigation Report

**Date**: 2026-03-09
**Script**: `inspect_image.sh` — raw data saved in `/tmp/inspect_25.11.01/`, `/tmp/inspect_26.02/`, `/tmp/inspect_myverl/`

**Legend**:
- **A** = `nvcr.io/nvidia/nemo:25.11.01` (official base, old)
- **B** = `nvcr.io/nvidia/nemo:26.02` (official base, new)
- **C** = `spacegoing/myverl:fixnvcc_mb_b479_vlmconfig_mjgym.dev` (your dev image, built on A)

---

## 1. System-Level Components

| Component | A (25.11.01) | B (26.02) | C (myverl) |
|-----------|-------------|-----------|------------|
| Python | 3.12.3 | 3.12.3 | 3.12.3 |
| CUDA (nvcc) | 13.0.88 | 13.0.88 | 13.0.88 |
| cuDNN | **9.13.1** | **9.18.0** | **9.13.1** (=A) |
| NCCL | **2.28.3** | **2.28.9** | **2.28.3** (=A) |
| GCC | 13.3.0 | 13.3.0 | 13.3.0 |
| CMake | 3.31.6 | 3.31.6 | 3.31.6 |
| Ubuntu | 24.04.3 | 24.04.3 | 24.04.3 |
| NVIDIA Build | 243847214 | **271855353** | 243847214 (=A) |

---

## 2. Core ML Frameworks

| Package | A (25.11.01) | B (26.02) | C (myverl) |
|---------|-------------|-----------|------------|
| **PyTorch** | 2.9.0a0 (nv25.09) | **2.10.0a0 (nv25.11)** | 2.9.0a0 (=A) |
| **torchvision** | 0.24.0a0 | **0.25.0a0** | 0.24.0a0 (=A) |
| **torchao** | — | 0.14.0+git | 0.13.0+git |
| **torch_tensorrt** | — | 2.10.0a0 | 2.9.0a0 |
| **triton** | 3.4.0 (`triton` pkg) | 3.5.0 (`pytorch-triton` pkg) | 3.4.0 (`triton`, =A) |
| **flash_attn** | 2.7.4.post1 | 2.7.4.post1+25.11 | 2.7.4.post1 (=A) |
| **flashinfer** | 0.5.3 | 0.5.3 | 0.5.3 |
| **transformer_engine** | 2.9.0+70f53666 | **2.12.0+5671fd36** | 2.9.0+70f53666 (=A) |
| **nvfuser** | — | 0.2.34+gitfce1aec | 0.2.29+gita71c674 |

---

## 3. vLLM (CRITICAL)

| | A (25.11.01) | B (26.02) | C (myverl) |
|--|-------------|-----------|------------|
| **Version** | 0.10.1.1 (nv25.9) | **0.14.2.dev0** | **0.12.0+cu130** (custom) |
| **Install type** | dist-packages (non-editable) | dist-packages (non-editable) | **editable at /opt/vllm** |
| **Source dir** | `/opt/vllm/vllm-src/` | **NO /opt/vllm/** | `/opt/vllm/` (your COPY) |
| **Location** | `/usr/local/lib/.../dist-packages` | `/usr/local/lib/.../dist-packages` | `/opt/vllm/vllm/` |

In **B (26.02)**, there is no `/opt/vllm/` directory at all. vLLM 0.14.2 is installed as a regular package in dist-packages. The old Dockerfile's `rm -rf /opt/vllm` will be a no-op — you need `pip uninstall -y vllm` first.

---

## 4. NeMo Ecosystem (MAJOR STRUCTURAL CHANGE)

| Package | A (25.11.01) | B (26.02) | C (myverl) |
|---------|-------------|-----------|------------|
| **megatron-core** | 0.16.0rc0 (pip, editable @ `/opt/megatron-lm`) | **0.16.0** (importable, NOT in pip, source @ `/opt/Megatron-Bridge/3rdparty/Megatron-LM/`) | 0.16.0rc0 (=A) |
| **nemo-toolkit** | 2.6.0rc0 (pip, editable @ `/opt/NeMo`) | **2.7.0** (importable, NOT in pip, source @ `/opt/NeMo`) | 2.6.0rc0 (=A) |
| **NeMo-FW** | 25.11rc3 (pip, editable @ `/opt/NeMo-FW`) | present but **NOT in pip** | 25.11rc3 (=A) |
| **megatron-bridge** | 0.2.0 (pip, editable @ `/opt/Megatron-Bridge`) | **NOT in pip** (source at `/opt/Megatron-Bridge`) | 0.2.0 (=A) |
| **nvidia-modelopt** | 0.37.0 (editable @ `/opt/TensorRT-Model-Optimizer`) | 0.37.0 (dist-packages, non-editable) | 0.37.0 (=A) |
| **nvidia-resiliency-ext** | 0.4.1 (editable @ `/opt/nvidia-resiliency-ext`) | 0.4.1+cuda13 (dist-packages) | 0.4.1 (=A) |

In **26.02**, NeMo ecosystem packages are **NOT registered in pip metadata** but ARE importable via source paths. `megatron-core` now lives inside Megatron-Bridge as a 3rdparty submodule rather than standalone at `/opt/megatron-lm/`.

---

## 5. `/opt/` Directory Structure Changes (A vs B)

| Directory | A (25.11.01) | B (26.02) |
|-----------|-------------|-----------|
| `/opt/vllm/` | source (vllm-src, vllm-flash-attn) | **REMOVED** |
| `/opt/megatron-lm/` | megatron-core source | **REMOVED** (now inside Megatron-Bridge) |
| `/opt/xformers/` | xformers source | **REMOVED** |
| `/opt/nvidia-resiliency-ext/` | editable source | **REMOVED** |
| `/opt/TensorRT-Model-Optimizer/` | modelopt source | **REMOVED** |
| `/opt/rdma-core/` | not present | **NEW** |
| `/opt/DeepEP/` | present | present |
| `/opt/NeMo/` | present | present |
| `/opt/NeMo-FW/` | present | present |
| `/opt/Megatron-Bridge/` | present | present (now contains megatron-lm as 3rdparty) |

---

## 6. Package Install Location Change

In **A**, many packages split across two paths:
- `/opt/venv/lib/python3.12/site-packages/` (NeMo ecosystem, transformers, etc.)
- `/usr/local/lib/python3.12/dist-packages/` (torch, vllm, flash_attn, etc.)

In **B**, **nearly everything is in `/usr/local/lib/python3.12/dist-packages/`** except transformer_engine and a few others still in `/opt/venv/`. The Dockerfile symlink `ln -s /usr/local/lib/python3.12/dist-packages` is still valid.

---

## 7. wandb

| | A | B | C |
|--|---|---|---|
| wandb | 0.23.1 | **NOT INSTALLED** | 0.23.1 (=A) |

The old Dockerfile runs `wandb login` early. On B, you need to `pip install wandb` before that.

---

## 8. Packages in C (myverl) NOT in B (26.02) — Dockerfile adds these

**Dockerfile-added (need re-install on B)**: `gpustat`, `codetiming`, `tensordict`, `mathruler`, `pylatexenc`, `torchdata`, `setuptools-scm`, `anthropic`, `model-hosting-container-standards`, `wandb`, `mbridge`, `verl`, `mjnemogym`

**Were in A but removed from B (may need if your code uses them)**: `xformers`, `triton` (renamed to `pytorch-triton`), `megatron-core` (pip metadata), `wandb`, `hydra-core`

---

## 9. Key Version Diffs: C (myverl) vs B (26.02)

| Package | C (myverl) | B (26.02) |
|---------|-----------|-----------|
| torch | 2.9.0a0 | **2.10.0a0** |
| vllm | 0.12.0 | **0.14.2** |
| transformer_engine | 2.9.0 | **2.12.0** |
| ray | 2.52.1 | **2.54.0** |
| protobuf | 4.25.8 | **6.33.5** |
| compressed-tensors | 0.10.2 | **0.13.0** |
| openai | 2.11.0 | **2.21.0** |
| peft | 0.13.2 | **0.18.1** |
| setuptools | 79.0.1 | **81.0.0** |
| tensorrt | 10.13.3.9 | **10.14.1.48** |
| transformers | 4.57.3 | **4.56.0** (downgraded!) |
