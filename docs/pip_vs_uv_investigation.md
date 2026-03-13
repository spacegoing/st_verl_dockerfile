# Investigation: pip vs uv Dual Package Manager in nemo:26.02

**Date**: 2026-03-13
**Triggered by**: flashinfer version discrepancy (pip says 0.5.3, runtime reports 0.6.4)

---

## Executive Summary

**The nemo:26.02 base image uses a dual-layer package architecture: a uv-managed venv
at `/opt/venv/` (primary, higher priority) on top of system site-packages at
`/usr/local/lib/python3.12/dist-packages/` (secondary, lower priority).**

**Our Dockerfile uses `pip install` which writes to the LOWER priority location.
Any package that also exists in the uv-managed venv is SHADOWED at runtime — Python
imports the venv's version, not ours.**

**This means our `transformers==4.57.3` pin, `wandb`, and other critical installs
are partially or completely ineffective at runtime.**

---

## 1. Architecture of nemo:26.02

### Two Package Stores

The image has TWO Python package directories, both visible to the same Python interpreter:

```
HIGH PRIORITY (searched first):
  /opt/venv/lib/python3.12/site-packages/    (uv-managed, 911 entries)

LOW PRIORITY (searched second):
  /usr/local/lib/python3.12/dist-packages/   (system/pip-managed, 781 entries)
```

### Python Resolution

```
$ which python3
/opt/venv/bin/python3 → /opt/venv/bin/python → /usr/bin/python
```

The venv is created with `--system-site-packages`, so BOTH directories are in `sys.path`:

```python
sys.path = [
    ...
    '/opt/venv/lib/python3.12/site-packages',    # ← FIRST (uv packages win)
    ...
    '/usr/local/lib/python3.12/dist-packages',   # ← SECOND (pip packages lose)
    ...
]
```

### How NVIDIA Built It

From `docker history`, the 26.02 image was built using:

```bash
# Step 1: Install uv
curl -LsSf https://astral.sh/uv/0.7.2/install.sh | sh

# Step 2: Create venv with system site-packages access
uv venv /opt/venv --system-site-packages

# Step 3: Install NeMo-FW and all its dependencies via uv sync
cd /opt/NeMo-FW && uv sync --no-cache-dir --all-groups --inexact

# Step 4: Some packages (vllm, xgrammar, deep_ep) installed via pip
pip install /tmp/vllm/vllm*.whl
pip install --no-deps xgrammar==0.1.25
```

**Key ENV vars baked into the image**:
```
UV_CACHE_DIR=/opt/uv_cache
UV_VERSION=0.7.2
UV_LINK_MODE=copy
UV_PROJECT_ENVIRONMENT=/opt/venv
```

The venv config at `/opt/venv/pyvenv.cfg`:
```ini
home = /usr/bin
implementation = CPython
uv = 0.7.2
version_info = 3.12.3
include-system-site-packages = true    # ← THIS makes both stores visible
```

---

## 2. The Problem: pip Installs to the Wrong Location

### Where pip Writes

Despite `python3` being the venv python, `pip` is the **system pip** at `/usr/local/bin/pip`:

```
$ which pip
/usr/local/bin/pip
$ head -1 /usr/local/bin/pip
#!/usr/bin/python
```

And pip has a constraint that makes it install to the system location:
```
PIP_BREAK_SYSTEM_PACKAGES=1
```

**Result**: `pip install X` → writes to `/usr/local/lib/python3.12/dist-packages/`

### Where Python Reads

Python imports from `/opt/venv/lib/python3.12/site-packages/` FIRST.

**If a package exists in BOTH locations, the venv copy wins.**

### Proof: Our transformers Pin is Ineffective

```
$ pip show transformers
Location: /usr/local/lib/python3.12/dist-packages
Version: 4.57.3          ← Our Dockerfile pin

$ python3 -c "import transformers; print(transformers.__version__)"
4.57.6                    ← Runtime uses venv's version, NOT ours
```

---

## 3. Full Impact Assessment

### Packages Where Our pip install is SHADOWED (Ineffective)

These packages exist in BOTH locations. Python uses the venv version (column 3), not
what pip installed (column 2):

| Package | pip installed (to /usr/local) | Runtime uses (from /opt/venv) | Impact |
|---------|------------------------------|-------------------------------|--------|
| **transformers** | **4.57.3** (our pin) | **4.57.6** | **Our pin is ignored** |
| **wandb** | **0.25.1** (our install) | **0.24.0** | **Our version is ignored** |
| **immutabledict** | **4.3.1** (our install) | **4.2.0** | Our version is ignored |
| protobuf | 6.33.5 | 5.29.6 | Venv wins (probably correct for NeMo) |
| pydantic | 2.12.4 | 2.13.0b1 | Venv wins (beta!) |
| fastapi | 0.121.3 | 0.132.0 | Venv wins |
| flashinfer-python | 0.5.3 | 0.6.4 | Venv wins |
| transformer-engine | 2.9.0 | 2.12.0 | Venv wins (correct for 26.02) |
| nvidia-modelopt | 0.37.0 | 0.41.0 | Venv wins |
| numba | 0.61.2 | 0.64.0 | Venv wins |
| scipy | 1.16.3 | 1.17.1 | Venv wins |
| ... | ... | ... | 80+ total mismatches |

### Packages Where Our pip install WORKS (No Conflict)

These packages are ONLY in `/usr/local/` (not in the venv), so pip install is effective:

| Package | Version | Works? |
|---------|---------|--------|
| yappi | 1.7.3 | Yes (only in /usr/local) |
| gprof2dot | 2025.4.14 | Yes |
| pydot | 4.0.1 | Yes |
| math-verify | 0.9.0 | Yes |
| latex2sympy2-extended | 1.11.0 | Yes |
| openapi-schema-validator | 0.8.1 | Yes |
| pylatexenc | 2.10 | Yes |
| codetiming | 1.4.0 | Yes |
| mathruler | 0.1.0 | Yes |
| orjson | 3.11.7 | Yes |
| pyvers | 0.2.2 | Yes |
| tensordict | 0.11.0 | Yes |
| gpustat | 1.1.1 | Yes |
| mbridge | 0.15.1 | Yes |
| verl | 0.8.0.dev0 | Yes (editable, resolves via .pth) |
| nemo-gym | 0.2.0rc0 | Yes (editable, resolves via .pth) |
| vllm | 0.12.0 | Yes (editable at /opt/vllm, resolves via .pth) |
| verifiable-instructions | 0.1.0 | Yes |

### Packages Where Our pip install is PARTIALLY Shadowed

- **bitsandbytes**: pip installed 0.49.2 to /usr/local. Not in venv. **Works.**
- **hydra-core**: pip installed 1.3.2 to /usr/local. In venv as 1.3.2 too. **Works (same version).**
- **langdetect**: pip installed 1.0.9 to /usr/local. In venv as 1.0.9 too. **Works (same version).**
- **absl-py**: pip installed 2.3.1 to /usr/local. Venv has 2.4.0. **Venv wins.**

---

## 4. How the Official NCR Image Uses uv

### Build Process

The official image build at `/opt/NeMo-FW/` uses a `pyproject.toml` + `uv.lock` workflow:

1. **Early layers**: Install uv, create venv
2. **Mid layers**: pip installs for CUDA packages (vllm, xgrammar, deep_ep, tensorrt-llm)
   These go to `/usr/local/` (system site-packages)
3. **Late layers**: `uv sync --all-groups --inexact` from NeMo-FW's pyproject.toml
   This installs the full NeMo stack to `/opt/venv/` — including newer versions of packages
   that were already pip-installed in step 2

This creates the dual-layer architecture intentionally: NVIDIA uses uv for the main NeMo
ecosystem (with proper lockfile resolution), and pip for a few CUDA packages that need
`--no-build-isolation` or custom wheel paths.

### The `--inexact` Flag

`uv sync --inexact` means: install the lockfile's packages but don't remove packages
that aren't in the lockfile. This is why system site-packages survive — uv adds to the
venv without removing what pip put in `/usr/local/`.

### ENV Configuration

```bash
UV_CACHE_DIR=/opt/uv_cache       # uv cache directory
UV_VERSION=0.7.2                  # uv version used to build
UV_LINK_MODE=copy                 # copy files instead of symlinks
UV_PROJECT_ENVIRONMENT=/opt/venv  # target venv for uv sync
```

---

## 5. Consequences for Our Dockerfile

### What's Broken

1. **`transformers==4.57.3` pin is ignored at runtime** — Python sees 4.57.6 from venv.
   If verl truly needs 4.57.3 features specifically, this might work since 4.57.6 is newer.
   But if 4.57.6 introduces breaking changes, we'd never know until runtime failure.

2. **`wandb` version mismatch** — We installed 0.25.1, runtime uses 0.24.0 from venv.
   The `wandb login` in L5 may have written credentials for a different wandb version
   than what actually runs.

3. **`pydantic` is a BETA** — Runtime uses 2.13.0b1 from venv. This is concerning for
   Gym server Pydantic validation. (Though our Gym tests passed, so likely OK.)

4. **`protobuf` version confusion** — pip says 6.33.5, runtime uses 5.29.6. Our previous
   deps report incorrectly stated "protobuf: 4.25.8 → 6.33.5 (BREAKING)" — the runtime
   was actually on 5.29.6 the whole time.

### What's NOT Broken

1. **Packages only we add** (yappi, gprof2dot, math-verify, etc.) — these are NOT in
   the venv, so they correctly resolve from `/usr/local/`.

2. **Editable installs** (verl, Gym, vllm) — these use `.pth` files which are resolved
   before site-packages, so they always win regardless of the pip/uv split.

3. **numpy** — Both locations have 1.26.4. No conflict.

4. **torch** — Only in `/usr/local/` (pip-installed by NVIDIA). Works.

---

## 6. Recommended Fix

### Option A: Use `uv pip install` Instead of `pip install`

Replace all `pip install` in the Dockerfile with `uv pip install`:

```dockerfile
# Instead of:
pip install --no-cache-dir --no-deps transformers==4.57.3

# Use:
uv pip install --no-deps transformers==4.57.3
```

`uv pip install` installs to the **venv** (`/opt/venv/lib/python3.12/site-packages/`),
which is the HIGH priority location. This ensures our installs take effect at runtime.

**Caveats**:
- `uv pip install` supports `--no-deps` but NOT `--no-build-isolation`
- For CUDA builds (vLLM, grouped_gemm, etc.), we may still need pip with
  `--no-build-isolation`. These packages go to `/usr/local/` but they work because
  they're NOT in the venv (uv doesn't install CUDA extensions).

### Option B: Install to venv explicitly with pip

```dockerfile
pip install --no-cache-dir --no-deps --target /opt/venv/lib/python3.12/site-packages/ \
    transformers==4.57.3
```

Less clean, but avoids needing to figure out uv flags.

### Option C: Remove conflicting packages from venv

After pip install, delete the venv's copy of packages we're overriding:

```dockerfile
pip install --no-cache-dir --no-deps transformers==4.57.3 && \
rm -rf /opt/venv/lib/python3.12/site-packages/transformers*
```

This forces Python to fall through to our `/usr/local/` copy.

### Recommendation

**Option A** for pure-Python packages in L5 (wandb, transformers, etc.).
Keep `pip install --no-build-isolation` for CUDA builds in L3-L4 (vllm, grouped_gemm, etc.)
since those packages aren't in the venv anyway.

---

## 7. Actual Runtime Package Versions (Ground Truth)

These are what Python actually imports (from `/opt/venv/` unless noted):

| Package | Runtime Version | From |
|---------|----------------|------|
| torch | 2.10.0a0+nv25.11 | /usr/local (no venv copy) |
| vllm | 0.12.0 | /opt/vllm (editable .pth) |
| transformers | 4.57.6 | /opt/venv |
| numpy | 1.26.4 | /opt/venv |
| protobuf | 5.29.6 | /opt/venv |
| pydantic | 2.13.0b1 | /opt/venv |
| fastapi | 0.132.0 | /opt/venv |
| flashinfer | 0.6.4 | /opt/venv |
| transformer_engine | 2.12.0 | /opt/venv |
| wandb | 0.24.0 | /opt/venv |
| flash_attn | 2.7.4.post1 | /usr/local (no venv copy) |
| ray | 2.54.0 | (check needed) |
| verl | 0.8.0.dev0 | editable (.pth) |
| nemo-gym | 0.2.0rc0 | editable (.pth) |
| megatron-core | 0.16.0 | /opt/Megatron-Bridge/3rdparty |
| nemo | 2.7.0 | /opt/NeMo |
