# Dev Manual: Gym Migration (iter1)

**Started**: 2026-03-11
**Goal**: Replace MJ_NEMO_GYM standalone scoring package with official NemoGym (Gym)
**Image**: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`

---

## Migration Plan

### Overview

MJ_NEMO_GYM was a manually-extracted subset of Gym's resources server scoring
functions, packaged as a standalone library (`verl_compute_score`). Now that the
verl infra is stable, we migrate to the official Gym package.

**Key constraint**: Gym is designed as a server framework (FastAPI + Ray + aiohttp).
For verl RL training, we only need the **scoring/verification functions** (offline),
NOT the server infrastructure. We must install Gym with `--no-deps` and selectively
add only the deps our scoring code paths actually import.

### Task 1: verl Code Changes

#### 1A. dapo.py — Replace mjnemogym with Gym scoring

**Old** (`dapo.py:21`):
```python
from mjnemogym import verl_compute_score
```

**New** (`dapo.py:21`):
```python
from verl.experimental.reward_loop.reward_manager.gym_compute_score import verl_compute_score
```

#### 1B. Gym Scoring Adapter — gym_compute_score.py

Created `verl/verl/experimental/reward_loop/reward_manager/gym_compute_score.py`.
This adapter replaces MJ_NEMO_GYM by importing scoring logic from Gym's installed
packages. Same `verl_compute_score` interface.

**Import mapping per domain:**

| Domain | MJ_NEMO_GYM Import | Gym Import |
|--------|-------------------|------------|
| Math | `math_verify` (external lib) | `math_verify` (same) |
| Code | `mjnemogym.code_gen.lcb_integration` | `resources_servers.code_gen.lcb_integration` |
| MCQA | Pure regex in `mjnemogym.mcqa.score` | Pure regex (inlined in adapter) |
| IF | `mjnemogym.instruction_following.verifiable_instructions` (bundled) | `verifiable_instructions` (standalone package) |
| SO | `openapi_schema_validator` (external lib) | `openapi_schema_validator` (same) |
| WA | `mjnemogym.workplace_assistant.utils` | `resources_servers.workplace_assistant.utils` |

**Key differences from MJ_NEMO_GYM:**
- Code: Uses `resources_servers.code_gen.lcb_integration` instead of bundled copy
- IF: Uses standalone `verifiable-instructions` package instead of bundled copy
- WA: Uses `resources_servers.workplace_assistant.utils.is_correct` directly
- All others: Identical logic, same external libraries

#### 1C. rl_dataset.py — No changes needed

Current code (line 356-357) already handles JSON-serialized extra_info:
```python
if isinstance(row_dict["extra_info"], str):
    row_dict["extra_info"] = json.loads(row_dict["extra_info"])
```

The Nemotron blend dataset parquet (preprocessed by `preprocess_nemogym_blend_v5.py`)
uses this exact format. Fully compatible.

#### 1D. Data preprocessing

Existing `verl/plans/preproc_nemogym/preprocess_nemogym_blend_v5.py` already converts
NemoGym JSONL to verl parquet with the correct schema:
- `data_source`: Maps agent names to `nemogym_*` keys
- `prompt`: Chat messages as native list
- `reward_model`: `{"ground_truth": str, "style": "rule"}`
- `extra_info`: JSON-serialized domain-specific metadata

No changes needed to preprocessing.

### Task 2: Dockerfile Changes

#### 2A. Changes summary

1. **Remove** `COPY MJ_NEMO_GYM` → **Replace with** `COPY Gym`
2. **Remove** `pip install -e MJ_NEMO_GYM` → **Replace with** `pip install --no-deps -e Gym`
3. **Add** `COPY verifiable-instructions` + `pip install --no-deps verifiable-instructions`
4. **Add** to Layer 5: `langdetect`, `absl-py`, `immutabledict` (deps of verifiable-instructions)

#### 2B. Dependency analysis

**Gym's scoring code paths import:**
- `math_verify` — Already in L5
- `resources_servers.code_gen.lcb_integration` — Comes with Gym editable install
- `verifiable_instructions` — Standalone package from git (needs separate install)
- `openapi_schema_validator` — Already in L5
- `resources_servers.workplace_assistant.utils` — Comes with Gym editable install
- `pydantic` — Already in 26.02 base
- `pandas` — Already in 26.02 base (needed by WA tools)
- `nltk` — Already in 26.02 base (needed by verifiable_instructions)

**verifiable-instructions deps (not in base):**
- `langdetect` — Added to L5
- `absl-py` — Added to L5
- `immutabledict` — Added to L5

**NOT needed (server infra, skipped via --no-deps):**
- `openai`, `fastapi`, `uvicorn`, `uvloop`, `mlflow`, `yappi`, `gprof2dot`, `pydot`, `itsdangerous`, `devtools`

#### 2C. Protected packages check

Cross-reference with "DO NOT TOUCH" list from dockerfile_update_plan.md SA-4.
Gym does NOT depend on torch, flash_attn, or any CUDA packages. All pip installs
use `--no-deps`. **Safe.**

#### 2D. New Dockerfile layer structure

```
L10: COPY verl                                     (changes often)
L11: COPY Gym                                      (changes often)
L12: COPY verifiable-instructions                   (stable)
L13: RUN pip install -e verl + Gym + verifiable-instructions
L14: WORKDIR
L15: CMD
```

Total: 122 layers (107 base + 15 new), well under 127 limit.

### Task 3: Documentation

- This file (dev_manual_gym_migration.md) — living document
- Updated Dockerfile comments
- `gym_compute_score.py` has full docstring

---

## Scoring Function Comparison: MJ_NEMO_GYM vs Gym Adapter

| Domain | MJ_NEMO_GYM | Gym Adapter | Functional Equivalence |
|--------|-------------|-------------|----------------------|
| Math | Library-only (`math_verify`) | Library-only (`math_verify`) | Identical |
| Code | Sync `check_correctness` | Sync `check_correctness` from Gym's `lcb_integration` | Identical |
| MCQA | Pure regex | Pure regex (inlined) | Identical |
| IF | Bundled `verifiable_instructions` | Standalone `verifiable_instructions` package | Identical (same source) |
| SO | `openapi_schema_validator` + parquet bug fix | Same + parquet bug fix | Identical |
| WA | Bundled `utils.is_correct` | Gym's `utils.is_correct` | Identical |

---

## Bugs & Fixes Log

| # | Date | Phase | Issue | Root Cause | Fix |
|---|------|-------|-------|------------|-----|
| 18 | 03-11 | smoke | torch.compile AssertionError `expected size 3072==512` on B300 | torch inductor shape mismatch during vLLM warmup with torch.compile on sm_103 | `enforce_eager=True` in rollout config |
| 19 | 03-12 | build | `ModuleNotFoundError: No module named 'lcb_integration'` | Gym's `compute_code_generation_metrics.py` uses bare `from lcb_integration.X` imports that fail when imported as namespace package | Changed to relative imports (`from .pass_k_utils`, `from .testing_util`, `from .lm_styles`) in 3 files |
| 20 | 03-12 | build | `verl_compute_score` raises KeyError for `math_dapo` data_source | Adapter only had `nemogym_*` keys; existing DAPO datasets use `math_dapo` | Added fallback to `default_compute_score` for non-nemogym data sources |
| 21 | 03-12 | runtime | Ray job submit `ServerDisconnectedError` | Container started Ray with wrong `--node-ip-address=10.12.11.5` (b31) on b32; proxy intercepting Ray dashboard requests | Started container manually with correct IP and cleared proxy env vars |

---

## Changelog

| Step | Status |
|------|--------|
| Write migration plan | Done |
| Analyze Gym scoring code imports | Done |
| Create Gym scoring adapter (`gym_compute_score.py`) | Done |
| Edit dapo.py import | Done |
| Verify rl_dataset.py compatibility | Done (no changes needed) |
| Clone verifiable-instructions repo | Done |
| Dockerfile: replace MJ_NEMO_GYM with Gym | Done |
| Dockerfile: add verifiable-instructions | Done |
| Dockerfile: add langdetect/absl-py/immutabledict to L5 | Done |
| Download Nemotron blend dataset | Pending |
| Preprocess sample data | Pending |
| Build image | Done |
| Fix Gym lcb_integration imports (relative) | Done |
| Add fallback to default_compute_score | Done |
| Test: verify scoring with Gym | Done (import + unit test passed) |
| Test: end-to-end training (smoke) | Done (val + step 1 completed) |
| Final dep diff report | Pending |
