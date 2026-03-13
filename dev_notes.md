# Dev Notes — Detailed Changelog

All changes to the Docker image, Dockerfile, and project infrastructure.

---

## 2026-03-13: ng_run Gym Integration (Isolated Venv Architecture)

### What Changed

Refactored Dockerfile to give Gym its own isolated `.venv` with per-server venvs created
by `ng_run "+dry_run=true"`, replacing the old approach of installing Gym into `/opt/venv/`
with `--no-deps`.

**Design references**: All decisions based on `verl/plans/nemo_gym_worker/`:
- `README.md` §141-178: Dockerfile Guide (exact commands)
- `design_manual.md` §3-5: Dual Ray cluster, per-server venv isolation, Dockerfile layers
- `research/plan_uv_gym_integration.md`: uv >=0.9.30 requirement, migration checklist
- `scripts/start_gym.sh`: Runtime startup (port 6380, skip_venv_if_present=true)

### Dockerfile Changes

| Section | Before | After | Why (reference) |
|---------|--------|-------|-----------------|
| L5 | Included Gym-only packages: `math_verify`, `langdetect`, `openapi_schema_validator`, `latex2sympy2_extended`, `pyvers`, `absl-py`, `immutabledict`, `model_hosting_container_standards`, `anthropic` | Removed all Gym-only packages | design_manual.md §4-5: these belong in Gym's isolated .venv, not system venv |
| L10-12 | 3 COPYs: verl, Gym, verifiable-instructions | 2 COPYs: verl, Gym only | plan_uv_gym_integration.md: lc_fix branch pulls verifiable-instructions from `git+https://github.com/spacegoing/my_verifiable-instructions.git` via `instruction_following/requirements.txt` |
| L12 | (was part of L13) | Separate RUN: verl editable install into `/opt/venv/` | Cleaner separation: verl in system venv, Gym in isolated venv |
| L13 | `uv pip install --no-deps -e .` (Gym into /opt/venv/) | Full ng_run pipeline (see below) | README §141-178: Gym needs own .venv + per-server venvs |

**L13 detailed breakdown** (single RUN layer to minimize layer count):
1. `curl -LsSf https://astral.sh/uv/install.sh | sh` — upgrades uv to latest (base has 0.7.2, Gym needs >=0.9.30 per plan_uv_gym_integration.md)
2. `uv venv --python 3.12 .venv` — creates Gym's isolated venv
3. `uv pip install -e ".[dev]"` — installs Gym + all dev deps into .venv
4. `uv pip install langdetect openapi_schema_validator math_verify absl-py nltk immutabledict` — extra domain verifier deps (README §157)
5. `cp gym_env.yaml → Gym/env.yaml` — policy_model placeholder for ${policy_base_url} interpolation (design_manual.md §8)
6. `ng_run "+dry_run=true"` — creates 7 per-server venvs with proper isolation (~5 min on overlay disk, design_manual.md §4)
7. `ln -sf lcb_integration → /usr/local/.../lcb_integration` — code_gen @ray.remote workers need system-importable lcb_integration (README §167-169)

### .dockerignore Changes
- Added `Gym/.venv` — prevents host's main Gym venv from bloating build context
- Added `verifiable-instructions/` — no longer COPYed into image

### Layer Count
- Before: 107 (base) + 15 (custom) = 122
- After: 107 (base) + 15 (custom) = 122 (removed 1 COPY, added 1 RUN, net same)

### Key Decisions
1. **Gym NOT in /opt/venv/ anymore** — old approach installed Gym editable into system venv with `--no-deps`. New approach: Gym has its own `.venv` at `/root/myCodeLab/host/Gym/.venv/`, fully isolated. This matches Gym's official integration pattern (README Dockerfile Guide).
2. **uv upgrade only for Gym** — L1-L5 still use base image's uv 0.7.2 (from `/opt/venv/bin/uv`). The upgraded uv at `~/.local/bin/uv` is only used for Gym venv creation. No interference with system venv.
3. **No verifiable-instructions COPY** — lc_fix branch commit `91b1b51e` changed `instruction_following/requirements.txt` to use `git+https://github.com/spacegoing/my_verifiable-instructions.git`. ng_run handles this automatically during per-server venv creation.
4. **Gym-only packages removed from L5** — 9 packages moved from system venv to Gym venv. These are only needed by Gym's domain verifiers, not by verl training loop.
5. **verl install separated** — L12 installs verl editable into `/opt/venv/` (needed for training loop). L13 handles all Gym setup. Clean separation.

### Build Result

*(updated after build completes)*

---

## 2026-03-13: pip → uv pip Migration (Package Manager Fix)

### Problem Discovered

The nemo:26.02 base image uses a **dual-layer Python package architecture**:

| Priority | Location | Manager | Contents |
|----------|----------|---------|----------|
| HIGH (searched first) | `/opt/venv/lib/python3.12/site-packages/` | uv | 911 packages (NeMo ecosystem) |
| LOW (searched second) | `/usr/local/lib/python3.12/dist-packages/` | pip | 781 packages (system/CUDA) |

Our Dockerfile used `pip install`, which writes to the LOW priority location. Any package
that also exists in `/opt/venv/` is **shadowed** — Python imports the venv version, not ours.

**Concrete impact**:
- `transformers==4.57.3` pin → runtime used 4.57.6 from venv (our pin ignored)
- `wandb==0.25.1` install → runtime used 0.24.0 from venv (our version ignored)
- `immutabledict`, `absl-py` → similarly shadowed
- 80+ packages had version mismatches between `pip freeze` and actual runtime

**Root cause**: NVIDIA builds 26.02 using `uv sync --all-groups --inexact` from NeMo-FW's
`pyproject.toml`, which populates `/opt/venv/`. They use pip only for a few CUDA packages
that need `--no-build-isolation`. The dual architecture is intentional (lockfile resolution
for NeMo, manual pip for CUDA), but `pip install` from Dockerfile layers writes to the
wrong location.

### Fix Applied

Replaced ALL `pip install` with `uv pip install` throughout the Dockerfile.
`uv pip install` writes to `/opt/venv/` (HIGH priority), ensuring our installs take effect.

**Changes to `Dockerfile.ncr.26.02.mydev`**:

| Layer | Before | After | Why |
|-------|--------|-------|-----|
| L1 | `pip uninstall -y vllm && pip install setuptools_scm` | `pip uninstall -y vllm; uv pip uninstall vllm; uv pip install setuptools_scm` | Clean both locations; install to venv |
| L3 | `pip install --no-deps --no-build-isolation -e .` | `uv pip install --no-deps --no-build-isolation --no-cache -e .` | vLLM → venv |
| L4 | `pip install --no-deps --no-build-isolation grouped_gemm/...` | `uv pip install --no-cache --no-deps --no-build-isolation ...` | CUDA exts → venv |
| L5 | `pip install --no-cache-dir --no-deps wandb transformers==4.57.3 ...` | `uv pip install --no-cache --no-deps wandb transformers==4.57.3 ...` | All pure-Python → venv |
| L6 | `pip3 install --no-cache-dir --no-deps .` (mbridge) | `uv pip install --no-cache --no-deps .` | mbridge → venv |
| L9 | `ln -s .../dist-packages .../dist-packages` | `ln -s .../site-packages .../site-packages` | Symlink → correct runtime location |
| L13 | `pip3 install --no-build-isolation --no-deps -e .` | `uv pip install --no-build-isolation --no-deps -e .` | verl/Gym/verifiable-instructions → venv |

**Updated "DO NOT TOUCH" list** with correct runtime versions from `/opt/venv/`:
- `flashinfer`: 0.5.3 → 0.6.4 (was reporting pip version, not runtime)
- `nvidia-modelopt`: 0.37.0 → 0.41.0
- Added: `protobuf=5.29.6`, `pydantic=2.13.0b1`, `fastapi=0.132.0`

**Flag changes**: `--no-cache-dir` (pip) → `--no-cache` (uv pip's equivalent flag)

### Why 26.02 Has Both pip and uv (Design, Not Mistake)

NVIDIA's build sequence:
1. **pip** installs PyTorch, CUDA packages (vllm, xgrammar, deep_ep, tensorrt-llm) →
   these need `--no-build-isolation` to use the system CUDA/torch
2. **uv venv** created with `--system-site-packages` → sees both stores
3. **uv sync** from NeMo-FW's `pyproject.toml` + `uv.lock` → installs full NeMo stack
   to `/opt/venv/`, including newer versions of some packages from step 1
4. **pip** installs last overrides (also shadowed by venv!)

The dual architecture is **by design** (lockfile resolution for NeMo ecosystem, manual pip
for CUDA builds). However, NVIDIA's own last pip overrides in step 4 are also shadowed —
this appears to be a minor oversight in their build, not just our problem.

### Files Modified
- `Dockerfile.ncr.26.02.mydev` — all `pip install` → `uv pip install`
- `readme.md` — updated layer descriptions, added pip_vs_uv reference
- `docs/pip_vs_uv_investigation.md` — created, full investigation report
- `docs/nemo_2602_image_anatomy.md` — created, 303-layer build phase analysis
- `.dockerignore` — created, reduces build context from ~8GB to ~100MB

### Build Result (Image ID: 67430e1e806b)

Build completed successfully (~55 min total, of which ~30min is vLLM CUDA compilation).
All `uv pip install` commands worked, including `--no-build-isolation` for CUDA compilation
(vLLM, grouped_gemm, causal_conv1d, mamba_ssm).

Image size: 65.7GB (up from 56.9GB because `.dockerignore` was created after build started;
the 7GB Gym/cache and Gym/responses_api_models dirs were included in context. Future builds
with `.dockerignore` will be smaller and faster to send context).

**Runtime verification — key fixes confirmed:**

| Package | Old Image (pip) | New Image (uv pip) | Status |
|---------|----------------|-------------------|--------|
| **transformers** | 4.57.6 (pin ignored!) | **4.57.3** | **FIXED** — pin now effective |
| wandb | 0.24.0 (venv shadowed ours) | 0.24.0 | OK (no version pin, keeps venv version) |
| vllm | 0.12.0 | 0.12.0 | OK (editable) |
| torch | 2.10.0a0+nv25.11 | 2.10.0a0+nv25.11 | OK (untouched) |
| flash_attn | 2.7.4.post1 | 2.7.4.post1 | OK (untouched) |
| transformer_engine | 2.12.0 | 2.12.0 | OK (untouched) |
| numpy | 1.26.4 | 1.26.4 | OK (untouched) |
| protobuf | 5.29.6 | 5.29.6 | OK (venv version) |

**Editable installs verified:**
- verl 0.8.0.dev0 → `/root/myCodeLab/host/verl/` (.pth in /opt/venv/)
- nemo_gym 0.2.0rc0 → `/root/myCodeLab/host/Gym/` (.pth in /opt/venv/)
- vllm 0.12.0+cu130 → `/opt/vllm/` (.pth in /opt/venv/)

**CUDA extensions verified:** grouped_gemm, causal_conv1d, mamba_ssm all import correctly.

**Note on `--no-deps` without version pin:** `uv pip install --no-deps wandb` does NOT
upgrade an already-installed version. The venv already had 0.24.0, and without `--reinstall`
or a version pin like `wandb>=0.25`, uv considers it satisfied. Pin explicitly if needed.

**Symlink change:** `/root/myCodeLab/site-packages` now points to `/opt/venv/lib/python3.12/site-packages/`
(was `/root/myCodeLab/dist-packages` pointing to `/usr/local/lib/python3.12/dist-packages/`).

**Build warning (harmless):** uv emits a warning about `exclude-dependencies` in NeMo-FW's
`pyproject.toml` — this is a newer uv field not recognized by the image's uv 0.7.2. Does
not affect our installs.

### Impact on Previous Deps Report

The `image_deps_report_v2.md` used `pip freeze` which only shows `/usr/local/` packages.
Many version numbers in that report are wrong (don't match runtime). Key corrections:

| Package | Report Said (pip freeze) | Actual Runtime (uv/import) |
|---------|-------------------------|---------------------------|
| transformers | 4.57.3 | was 4.57.6, now **4.57.3** (fixed) |
| protobuf | 6.33.5 | 5.29.6 |
| flashinfer | 0.5.3 | 0.6.4 |
| wandb | 0.25.1 | 0.24.0 |
| pydantic | 2.12.4 | 2.13.0b1 |
| fastapi | 0.121.3 | 0.132.0 |
| transformer_engine | 2.9.0 | 2.12.0 |
| nvidia-modelopt | 0.37.0 | 0.41.0 |

---

## 2026-03-12: Gym Server Migration (iter2)

### What Changed
- Migrated from MJ_NEMO_GYM (iter1, direct function calls) to official NemoGym (iter2, FastAPI servers)
- Added 6 Gym resource server domains: code_gen, mcqa, instruction_following, structured_outputs, workplace_assistant, math_with_judge
- Created blend dataset with all 6 domains (93,244 samples in train_v2.parquet)
- Math domain required HF data patching (22,056 rows from DAPO-Math-17k + Skywork-OR1-RL-Data)

### Files Added/Modified
- `Dockerfile.ncr.26.02.mydev` — added Gym, verifiable-instructions COPY+install layers
- `verl/my_scripts/gym_server_runner.py` — standalone server launcher
- `verl/my_scripts/launch_gym_servers.sh` — multi-server launcher with health checks
- `verl/verl/workers/reward_manager/nemogym_server.py` — multi-domain reward manager
- `verl/my_scripts/run_moonlight_1node_blend_smoke.sh` — updated for blend dataset + server mode
- `dev_manual_iter2_gym_server.md` — full implementation plan and bug log

### Why Server Mode
- Standard `/verify` HTTP interface across all domains
- Extensible (new domains = new servers, not code changes)
- Official NemoGym pattern (follows verl/examples/tutorial/nemo_gym/)
- Scalable to 64-node jobs (each pod runs local servers)

---

## 2026-03-11: Initial Image Build (nemo:26.02 base)

### What Changed
- Created Dockerfile.ncr.26.02.mydev based on nemo:26.02 (previously nemo:25.11.01)
- Built vLLM 0.12.0 from source with CUDA compilation for B300 (sm_103)
- Added CUDA extensions: grouped_gemm, causal_conv1d, mamba_ssm (removed in 26.02)
- Set up 2-node docker-compose cluster (b31 head + b32 worker)
- Created dev_manual with full setup documentation

### Key Design Decisions
- `DOCKER_BUILDKIT=0`: Legacy builder required due to base image having ~107 layers (127 limit)
- `--no-deps` everywhere: Protects base image's carefully tuned package versions
- `--no-build-isolation` for CUDA builds: Must use system torch/CUDA, not isolated copies
- Editable installs for verl/Gym: Live development via docker-compose mount overlay
- vLLM at `/opt/vllm/` (not under mount): Immutable at runtime (CUDA compiled)

### B300-Specific Workarounds
- `VLLM_ATTENTION_BACKEND=CUTLASS_MLA`: B300 reports sm_103, but vLLM's `is_device_capability(100)` does exact match → forced backend selection
- NCCL RoCE config: `NCCL_IB_GID_INDEX=3` for RoCEv2 routable GID
- NVSHMEM GPU-initiated RDMA for DeepEP MoE expert dispatch
