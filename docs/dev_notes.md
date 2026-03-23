# Dev Notes — Detailed Changelog

All changes to the Docker image, Dockerfile, and project infrastructure.

---

## 2026-03-20/21: 2-Node Training — Debug & Fix (EP=16 GRPO, Moonlight-16B)

### Goal

Get 2-node Megatron GRPO training working on b32 (head, 10.12.11.6) + b31 (worker, 10.12.11.5)
with EP=16 across 16 GPUs total. Script: `verl/my_scripts/run_moonlight_2node_blend_smoke.sh`.

Two real bugs (NCCL GID, DeepEP NVSHMEM) plus two misdiagnosed bugs (Bugs 2/3, corrected 2026-03-21).

---

### Bug 1: NCCL `ibv_modify_qp` errno 22 on mlx5_14 (b31)

**Symptom:**
```
ibv_modify_qp failed with 22 Invalid argument on dev mlx5_14:1, local GID index 3,
local GID fe80::8c3d:8ff:fe64:7719
```

**Root cause:**

`NCCL_IB_GID_INDEX=3` selects GID slot 3 for RoCEv2 (the IPv4-mapped routable address
`::ffff:100.86.x.x`). On b32, mlx5_14's GID[3] is routable. But on **b31**, mlx5_14's GID[3]
is `fe80::` (link-local — not routable cross-node). GID[4] on b31's mlx5_14 is routable, but
we don't set GID_INDEX=4. All other NICs (mlx5_10-13, mlx5_15-17) have routable GID[3] on
both nodes.

Diagnosis: checked GID tables directly via
`/sys/class/infiniband/<nic>/ports/1/gids/<index>` on both nodes.

**Fix:** Exclude mlx5_14 from `NCCL_IB_HCA` in both places it's set:

`docker-compose.yml`:
```yaml
NCCL_IB_HCA: "mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_15:1,mlx5_16:1,mlx5_17:1"
```

`verl/my_scripts/my_deepep_env.yaml`:
```yaml
NCCL_IB_HCA: "mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_15:1,mlx5_16:1,mlx5_17:1"
```

Note: `my_deepep_env.yaml` is the Ray runtime env — it propagates to all Ray workers on all
nodes including b31. Both files must agree.

---

### Bugs 2 & 3: Misdiagnosis — Gym servers not started on b31 (corrected 2026-03-21)

**Original misdiagnosis:** Bugs 2 and 3 were initially diagnosed as separate issues (reward URL
must use b32's IP; Gym servers must bind to 0.0.0.0). Both diagnoses were wrong.

**Actual root cause:** The 2-node run script only started Gym servers on b32 (head).
`RewardLoopWorker` actors are scheduled round-robin on all alive verl Ray nodes (see
`reward_loop.py:248-260`), so workers land on both b31 and b32. Each worker uses the same
configured `server_url` (e.g. `localhost:20001`). Workers on b31 correctly try `localhost:20001`
on their own node — but b31 had no Gym servers running, causing connection refusal.

**Correct architecture:**
- Each node in the verl Ray cluster runs its own local Gym cluster (isolated Ray on port 6380)
- Gym servers bind to `localhost`; workers connect to `localhost:2000x`
- This matches the 1-node design (e.g. `run_moonlight_1node_blend_smoke.sh` uses `localhost` correctly)
- The initial workarounds (explicit b32 IP + 0.0.0.0 binding) masked the real issue and only
  worked by routing all reward calls to b32's workers, ignoring b31's workers entirely

**Confirmed with `ray status --address=127.0.0.1:6380`:** Each node's Gym Ray is a completely
independent 1-node cluster with its own port 6380. Verl Ray cluster on port 6379 is 2-node.
Four clusters total (2 Gym + 1 Verl on each node = 2 Gym + 1 Verl cross-node). All isolated.

**Real fix (2026-03-21):**

1. **`docker-compose.yml`**: Added `/root/.ssh:/root/.ssh:ro` volume mount so the container can
   use pdsh to reach b31.

2. **`run_moonlight_2node_blend_smoke.sh`**: Starts Gym on both nodes in parallel, uses `localhost`:
```bash
# Both start in background, script waits for both to be healthy
pdsh -w b31 "docker exec vrl bash ${GYM_SCRIPT}" &
bash "${GYM_SCRIPT}" &
```
Reward URLs reverted back to `localhost:2000x` (same as 1-node script).

3. **`start_gym_uv.sh`**: The `default_host: "0.0.0.0"` auto-patch is kept (harmless, useful for
   external monitoring), but is no longer required for correctness.

**Verified (2026-03-21):**
```
b32: [init] All 7 Gym servers healthy.   ← b32 gym running
b31: [init] All 7 Gym servers healthy.   ← b31 gym running
b32 Gym Ray (6380): 1 node, 8 GPU       ← b32 isolated gym cluster
b31 Gym Ray (6380): 1 node, 8 GPU       ← b31 isolated gym cluster
Verl Ray (6379):    2 nodes, 16 GPU     ← cross-node training cluster
```

---

### Bug 4: DeepEP/NVSHMEM flex dispatcher fails cross-node

**Symptom:**
```
socketProgress: Connection closed by remote peer b31<42864>
allgather of ipc handles failed
nvshmem initialization failed, exiting
Worker unexpectedly exits with a connection error code 2.
```

**Root cause:** `moe_token_dispatcher_type=flex` (DeepEP) requires NVSHMEM for cross-node
expert routing. NVSHMEM uses a UID socket bootstrap to establish IB RDMA connections
between nodes. This bootstrap fails on this cluster — the UID socket connection from b32
is refused/closed by b31 before IB handles can be exchanged.

**Fix:** Switched MoE dispatcher from `flex` (DeepEP/NVSHMEM) to `alltoall` (NCCL-based):

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
```

`alltoall` uses NCCL all-to-all for expert token routing — works over the same IB RoCEv2 links
(mlx5_10-13, mlx5_15-17) that are already working for NCCL. Performance is slightly lower than
DeepEP flex but fully functional cross-node.

**Known limitation:** DeepEP `flex` dispatcher (NVSHMEM IB bootstrap) not working on this
cluster. Root cause not fully diagnosed — likely requires NVSHMEM IB bootstrap configuration
(`NVSHMEM_BOOTSTRAP=IB` or proper UID socket routing between nodes).

---

### Ray GCS session mismatch (encountered during debugging)

During aggressive debugging, killing processes on ports 20001-20007 with `fuser` also killed
Ray head's gcs_server/raylet (they happened to be using ports in that range). This left the
Ray cluster in a broken state with orphaned GCS state at `/tmp/ray_gym`.

**Fix:** Full container recreate:
```bash
docker rm -f vrl
docker compose up -d head   # on b32
docker rm -f vrl
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker  # on b31
```

Lesson: Do not use `fuser -k` on port ranges. Kill processes by PID or name to avoid
accidentally killing Ray infrastructure.

---

### Working Configuration (as of 2026-03-20)

**Parallelism:** `NNODES=2`, `EP=16`, `gen_tp=1`, `train_tp=1`, `train_pp=1`

**MoE dispatch:** `alltoall` + `moe_enable_deepep=False` (not flex/NVSHMEM)

**NCCL NICs:** `mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_15:1,mlx5_16:1,mlx5_17:1`
(mlx5_14 excluded — b31 GID[3] is link-local)

**Reward URLs:** `http://localhost:2000x` (each node hits its own local Gym)

**Gym startup:** both nodes in parallel — `run_moonlight_2node_blend_smoke.sh` pdsh-starts b31 gym and starts b32 gym concurrently, waits for both

**Gym bind:** `default_host: "0.0.0.0"` — auto-patched by `start_gym_uv.sh` (harmless, kept for external monitoring)

**Training launch:**
```bash
# On b32
cd /mnt/public/lichang93/st_verl_dockerfile
docker rm -f vrl 2>/dev/null || true
docker compose up -d head

# On b31
docker rm -f vrl 2>/dev/null || true
docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker

# Back on b32: verify 2 nodes visible
docker exec vrl ray status

# Start Gym + training (handles gym config patch automatically)
docker exec vrl bash /root/myCodeLab/host/verl/my_scripts/run_moonlight_2node_blend_smoke.sh
```

**Step 1 metrics (first successful 2-node run):**
```
actor/pg_loss: 0.1501 → 0.0945 (step 32)
actor/grad_norm: 10.59
perf/max_memory_allocated_gb: 199.4 GB (b32, ~200 GB per node)
perf/throughput: 7.99 tokens/s (step 1), 11.36 tokens/s (step 2)
timing_s/step: 281s
```

Val acc at step 0 (baseline before any training):
```
workplace: 0.143, mcqa: 0.037, if: 0.038, code: 0.000, structured: 0.000
```

---

### Files Changed (2-node fix — cumulative)

| File | Change |
|------|--------|
| `docker-compose.yml` | `NCCL_IB_HCA`: removed `mlx5_14`; added `/root/.ssh:/root/.ssh:ro` volume mount |
| `verl/my_scripts/my_deepep_env.yaml` | `NCCL_IB_HCA`: removed `mlx5_14` |
| `verl/my_scripts/run_moonlight_2node_blend_smoke.sh` | Gym startup: pdsh starts b31 gym in parallel; reward URLs: `localhost:2000x`; dispatcher: `flex`→`alltoall`, `moe_enable_deepep`: `True`→`False` |
| `verl/my_scripts/start_gym_uv.sh` | Auto-patch `gym_blend_servers.yaml` with `default_host: "0.0.0.0"` (idempotent); section numbering 2→2, 3→3, 4→4 |
| `CLAUDE.md` | Added §6.2 (2-node training), §10 entries for bugs |

---

## 2026-03-20: Fix Docker Push — Switch to `docker buildx build --push`

### Problem

`docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` failed:

```
You're trying to push a manifest list/index which references multiple platform specific manifests,
but not all of them are available locally or available to the remote repository.
NotFound: content digest sha256:... not found
```

Even adding `--platform linux/amd64` to build and/or push did not fix it.

### Root Cause (full diagnosis)

Docker 29 with **containerd snapshotter** (`io.containerd.snapshotter.v1`) uses two separate
storage mechanisms:

- **Overlay2 snapshots**: layer filesystems for running containers (`/var/lib/docker/overlay2/`)
- **Containerd content store**: OCI blobs (layer tarballs) needed for registry push/pull

When building with `DOCKER_BUILDKIT=0`, Docker uses the legacy builder which stores layers
**only as overlay2 snapshots**. The OCI blobs are NOT written to the content store.

When `docker push` runs, it reads blobs from the content store (not overlay2). 85 out of 107
nvcr.io base layers were missing from the content store → push fails with "content digest not
found". The "manifest list" wording in the error is misleading; the actual issue is missing blobs.

`--platform linux/amd64` on build or push does not help because the problem is missing blob data,
not the manifest structure.

### Fix: `docker buildx build --provenance=false --sbom=false --push`

Two flags are critical:

**`--provenance=false --sbom=false`** (the real CCR-compatibility fix)
Without these, BuildKit adds an attestation manifest (SBOM/provenance) as an `unknown/unknown`
platform entry. This turns every pushed image into a manifest list. Old CCRs reject it.
With these flags: clean single-platform OCI image manifest — no manifest list, works everywhere.

**`--push`** (fixes the Docker 29 overlay2 blob issue)
BuildKit writes all layer blobs to the OCI content store during build, then uploads them directly.
`DOCKER_BUILDKIT=0` writes layers to overlay2 only; `docker push` reads from the content store
and can't find them → "content digest not found".

```bash
# Base image (proxy needed: nvcr.io FROM pull + apt-get in RUN):
PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128
NO_PROXY=registry.cn-hangzhou.aliyuncs.com

HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.base \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --network host \
  --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.base \
  --build-arg HTTP_PROXY=$PROXY \
  --build-arg HTTPS_PROXY=$PROXY \
  .

# Dev image (proxy also needed: curl astral.sh for uv upgrade + git clone in ng_run dry_run):
# PyPI packages use Aliyun mirror (UV_DEFAULT_INDEX ARG) — fast, no proxy needed for pip.
HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY NO_PROXY=$NO_PROXY \
docker buildx build -f Dockerfile.ncr.26.02.mydev \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --network host \
  --push \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
  --build-arg HTTP_PROXY=$PROXY \
  --build-arg HTTPS_PROXY=$PROXY \
  .

# Re-tag locally for docker-compose:
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
docker tag  registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
            myverl:ncr2602_vllm012.dev
```

**Why HTTPS_PROXY + NO_PROXY**: proxy needed for nvcr.io/apt; Aliyun auth fails through proxy →
`NO_PROXY` sends the `--push` direct to Aliyun.

### Layer cache behavior (BuildKit vs BUILDKIT=0)

BuildKit uses **content-addressed** caching:
- Cache key = hash(instruction text) + hash(COPYed file contents) + parent layer digest
- Layer is rebuilt **only if its content actually changed** — mtime/timestamp changes are ignored
- Much more reliable than BUILDKIT=0's mtime-based checking

`BUILDKIT=0` was previously used to avoid the provenance manifest list issue. `--provenance=false`
is the correct targeted fix — get BuildKit's better caching without the CCR-incompatible metadata.

---

## 2026-03-19: Aliyun PyPI Mirror + code_gen Ray Worker Fix

### PyPI mirror for fast downloads

Added `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` (as `ARG`/`ENV`) to both
`Dockerfile.base` and `Dockerfile.ncr.26.02.mydev`.

**Why**: Benchmarked corporate proxy throughput during docker build at ~33-96 KB/s.
At that speed `ray` (69.6MB) alone takes ~12-15 min. Direct mainland access to
`mirrors.aliyun.com` is significantly faster. Since `UV_DEFAULT_INDEX` is a well-known
uv env var, it applies to all `uv pip install` commands without per-command flags.
Can be overridden at build time: `--build-arg UV_DEFAULT_INDEX=https://pypi.org/simple/`

**Proxy is still needed** for: `curl` (uv self-install from astral.sh), `git clone`
(cutlass in vLLM build, verifiable-instructions in ng_run). Pass proxy for these via
`--build-arg HTTP_PROXY=...` (the official Docker way — predefined proxy build args are
automatically available in all RUN commands without `ARG` declarations in the Dockerfile).

### code_gen Ray worker fix: PYTHONPATH for lcb_integration

**Problem**: `check_correctness_remote` in `lcb_integration/compute_code_generation_metrics.py`
had no `runtime_env` on `@ray.remote`. The previous workaround was `ln -sf lcb_integration →
/usr/local/lib/python3.12/dist-packages/lcb_integration` so system Python could import it.
This was wrong for two reasons:
1. Ray workers should use the code_gen per-server venv (not system Python), for isolation
2. `lcb_integration` has no `pyproject.toml`, so it can't be pip-installed — it's a plain directory

**Root cause**: Ray workers spawn in a temp working directory. The server's `sys.path[0]`
(the `code_gen/` dir) is NOT inherited by workers. So even with the correct venv, workers
can't find `lcb_integration` unless its parent directory is on their `PYTHONPATH`.

**Fix** (two parts):
1. `py_executable: sys.executable` — workers use the code_gen venv's Python (not system Python)
2. `env_vars: {PYTHONPATH: _CODE_GEN_DIR}` — adds `code_gen/` to workers' Python path so
   `import lcb_integration` resolves

```python
_CODE_GEN_DIR = str(Path(__file__).parent.parent)  # evaluates to .../code_gen/ at import time
@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": _CODE_GEN_DIR},
    },
)
def check_correctness_remote(...):
```

`_CODE_GEN_DIR` is computed from `Path(__file__)` (path of `compute_code_generation_metrics.py`)
→ `.parent` = `lcb_integration/` → `.parent` = `code_gen/`. Evaluated at module import time
(when the server starts), so it always resolves to the real live path in the running container.

Pattern: `env_vars.PYTHONPATH` in `runtime_env` is the standard way to make non-installed
packages available to Ray workers. See also: `swerl_gen/eval/singularity_utils.py`.

Removed the `ln -sf` workaround from `Dockerfile.ncr.26.02.mydev` (Layer 13 RUN block).
Removed the symlink block from `start_gym_uv.sh`.

### Container test checklist

After each dev image rebuild, verify with:

```bash
docker run --rm --network host \
  -v /mnt/public/lichang93/st_verl_dockerfile:/root/myCodeLab/host \
  registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
  bash -c '
echo "=== Check 1: Gym venvs exist at /opt/ ==="
ls /opt/gym_venvs/ && ls /opt/gym_venvs/resources_servers/

echo "=== Check 2: Main venv Python ==="
/opt/gym_venvs/main/bin/python --version

echo "=== Check 3: ng_run available ==="
/opt/gym_venvs/main/bin/ng_run --help 2>&1 | head -3

echo "=== Check 4: gym_config YAMLs in /opt/ ==="
ls /opt/gym_config/

echo "=== Check 5: code_gen venv can import lcb_integration ==="
CODE_GEN_VENV=$(ls /opt/gym_venvs/resources_servers/ | grep code_gen | head -1)
/opt/gym_venvs/resources_servers/$CODE_GEN_VENV/.venv/bin/python -c "
import sys; sys.path.insert(0, \"/root/myCodeLab/host/Gym/resources_servers/code_gen\")
import lcb_integration; print(\"OK:\", lcb_integration.__file__)"

echo "=== Check 5b: _CODE_GEN_DIR resolves correctly ==="
/opt/gym_venvs/resources_servers/$CODE_GEN_VENV/.venv/bin/python -c "
import sys; sys.path.insert(0, \"/root/myCodeLab/host/Gym/resources_servers/code_gen\")
from lcb_integration.compute_code_generation_metrics import _CODE_GEN_DIR
import os; print(\"_CODE_GEN_DIR:\", _CODE_GEN_DIR)
print(\"lcb_integration dir exists:\", os.path.isdir(os.path.join(_CODE_GEN_DIR, \"lcb_integration\")))"

echo "=== Check 6: verl installed in system venv ==="
python -c "import verl; print(verl.__file__)"

echo "=== Check 7: env.yaml at Gym root ==="
ls /root/myCodeLab/host/Gym/env.yaml
'
```

Expected: all checks print OK/paths with no errors.

---

## 2026-03-19: Dockerfile Base/Dev Split + Gym Venv Decoupling from PFS

### What Changed

Two major changes bundled in this rebuild:

**1. Dockerfile split into base + dev images**

Split single `Dockerfile.ncr.26.02.mydev` into two files:
- `Dockerfile.base` (Layers 1-9: vLLM CUDA build, CUDA extensions, system deps) → tagged `ncr2602_vllm012.base`
- `Dockerfile.ncr.26.02.mydev` now starts `FROM ncr2602_vllm012.base` with only Layers 10-15 (verl + Gym)

**Why**: vLLM CUDA build (Layer 3) clones `nvidia/cutlass` from GitHub (~45MB). Connection
through proxy drops consistently. With the old single Dockerfile, ANY code change (even a
1-line verl fix) required surviving this network-sensitive step. The split means the base
image is built once and pushed; dev rebuilds start FROM it and never touch CUDA compilation.

Dev rebuild time: ~10-15 min (was ~30 min cold, failed 3x at cutlass git clone).

**2. Gym venvs moved from PFS → /opt/gym_venvs/ (node-local disk)**

- Before: Gym main venv at `Gym/.venv` (inside PFS mount at `/root/myCodeLab/host/`)
- After: Gym main venv at `/opt/gym_venvs/main/`; per-server venvs at `/opt/gym_venvs/resources_servers/<name>/.venv/`

**Why**:
- **Mount override**: docker-compose mounts PFS project dir onto `/root/myCodeLab/host/`, which wipes
  any venvs baked there. Venvs at `/opt/` survive the mount.
- **PFS slowness**: quarkfs random file I/O is slow. Python import from PFS is ~5-10x slower than
  from node-local `/opt/`. Moving venvs from PFS → local disk dramatically improves server startup.

ng_run flag `+uv_venv_dir=/opt/gym_venvs` is the key — ng_run respects this for both dry_run
(image build) and live startup, ensuring symmetry between image-baked paths and runtime paths.

Runtime startup time with pre-baked venvs: ~18s (skip_venv_if_present=true, no reinstall).

### Dockerfile Changes

| File | Before | After |
|------|--------|-------|
| `Dockerfile.base` | (did not exist) | New file: Layers 1-9 from old Dockerfile |
| `Dockerfile.ncr.26.02.mydev` | FROM nvcr.io/nvidia/nemo:26.02, 15 layers | FROM ncr2602_vllm012.base, 6 layers |
| `Dockerfile.patch_gym_venv` | Emergency fallback (FROM existing dev, rebuild venvs) | Deleted — superseded by base/dev split |
| L13 venv path | `Gym/.venv` | `/opt/gym_venvs/main` |
| L13 ng_run | no `+uv_venv_dir` | `+uv_venv_dir=/opt/gym_venvs` |

### .dockerignore Changes
- Added `Gym/.venv_*` — excludes stale test venvs (`.venv_test`, `.venv_test2`, etc.) that had
  transient build artifacts causing "file not found in context" errors during docker build context scan

### start_gym_uv.sh Changes
- `GYM_VENV_DIR=/opt/gym_venvs` (new var)
- `GYM_RAY`: `${GYM_DIR}/.venv/bin/ray` → `${GYM_VENV_DIR}/main/bin/ray`
- ng_run binary: `.venv/bin/ng_run` → `/opt/gym_venvs/main/bin/ng_run`
- Added `+uv_venv_dir=${GYM_VENV_DIR}` to ng_run call

### Build Timings (from docker history)
- COPY verl: ~1m08s (575MB — PFS read slow)
- COPY Gym: ~35s (27.8MB)
- verl install: ~35s
- Layer 13 (Gym venvs): ~18 min first time (uv cache cold); ~1-2 min warm
- Per-server venv installs during build: 850-978ms each (7 servers, uv cache warm on overlay fs)

### Images Pushed
- `ncr2602_vllm012.base`: 55.3GB
- `ncr2602_vllm012.dev`: 69.9GB

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

### Bugs & Fixes — Complete List

All bugs encountered during the ng_run Gym integration, ordered chronologically.

#### Build-Time Bugs (Dockerfile)

**Bug 1: `egg_base` directory missing** (build attempt 1)
```
error: error in 'egg_base' option: 'cache' does not exist or is not a directory
```
- **Cause**: Gym's `pyproject.toml` line 299: `egg_base = "cache"`. But `.dockerignore` excluded `Gym/cache/`.
- **Fix**: Added `mkdir -p cache` before `uv pip install -e ".[dev]"` in L13.
- **Reference**: `.dockerignore` still excludes `Gym/cache/` to save context size; we create an empty one in-image.

**Bug 2: `responses_api_models/vllm_model` missing** (build attempt 2)
```
RuntimeError: Missing pyproject.toml or requirements.txt for uv venv setup in server dir:
  /root/myCodeLab/host/Gym/responses_api_models/vllm_model
```
- **Cause**: `.dockerignore` had `Gym/responses_api_models/` which excluded the entire directory. ng_run's `policy_model` config (gym_blend_servers.yaml line 9-16) references `responses_api_models/vllm_model/` which needs its `pyproject.toml`.
- **Fix**: Removed `Gym/responses_api_models/` from `.dockerignore`.

**Bug 3: `pyvers` module not found** (build attempt 3 — image built, runtime verification failed)
```
ModuleNotFoundError: No module named 'pyvers'
```
- **Cause**: Incorrectly removed `pyvers` from L5, thinking it was Gym-only. But `tensordict/utils.py` imports `from pyvers import get_backend, ...` — it's a verl training dependency, not Gym.
- **Fix**: Added `pyvers` back to L5 install list.
- **Lesson**: Only remove packages you're certain are Gym-only. `pyvers` is used by `tensordict`, which is a core verl dependency.

#### Runtime Bugs (start_gym_uv.sh / container)

**Bug 4: "No available ports" — Gym Ray worker port range too small**
```
Invalid: No available ports. Please specify a wider port range using
--min-worker-port and --max-worker-port
```
- **Cause**: `--max-worker-port=7499` gave only 100 ports (7400-7499). ng_run spawns multiple worker processes per server, and with 7 servers the 100-port pool was exhausted.
- **Fix**: Changed `--max-worker-port` from 7499 to 7999 (600 ports).
- **Files changed**: `start_gym_uv.sh`, `plans/nemo_gym_worker/scripts/start_gym.sh`

**Bug 5: "Too many open files" — raylet crash (SIGABRT)**
```
Unhandled exception: open: Too many open files
```
- **Cause**: `--num-cpus=256` caused Ray to pre-create file descriptors proportional to CPU count, exceeding the container's default `ulimit -n 1024`.
- **Fix (two-part)**:
  1. Reduced `--num-cpus` from 256 to 32 in `start_gym_uv.sh`
  2. Added `nofile: soft: 65536, hard: 65536` to `docker-compose.yml` ulimits
- **Also removed**: `--memory=$((3900 * 1024 * 1024 * 1024))` (unnecessary, let Ray auto-detect)
- **Files changed**: `start_gym_uv.sh`, `plans/nemo_gym_worker/scripts/start_gym.sh`, `docker-compose.yml`

**Bug 6: Ray version mismatch — per-server venvs vs Gym main .venv**
```
RuntimeError: Version mismatch: The cluster was started with: Ray: 2.52.1 ...
This process on node was started with: Ray: 2.54.0
```
- **Cause**: The local per-server venvs (on host filesystem, visible via mount overlay) were built at a different time and had Ray 2.54.0, while the Gym main `.venv` (from Docker image) had Ray 2.52.1. When `skip_venv_if_present=true`, ng_run uses whatever venvs exist, and the mount overlay makes host venvs visible.
- **Fix**: Extracted correct per-server venvs from Docker image via `docker cp` to local filesystem, so the mount overlay exposes venvs matching the image's Ray version.
- **Root cause**: Docker-compose mount overlay hides image venvs; local stale venvs take precedence.

**Bug 7: Stale Ray GCS session assertion**
```
AssertionError: Session name session_2026-03-13_14-19-16_... does not match
persisted value b'session_2026-03-13_14-11-02_...'
```
- **Cause**: Leftover GCS server processes from previous failed `ray start` attempts. The old GCS still held the session in `/tmp/ray_gym/`, so new `ray start --head` failed session assertion.
- **Fix**: Kill stale GCS processes (`pkill -9 -f gcs_server`) and `rm -rf /tmp/ray_gym`. Ultimately fixed by recreating the container fresh.
- **Lesson**: After repeated Ray start/stop in the same container, stale state accumulates. Recreating the container is the cleanest fix.

**Bug 8: Ports already in use (TIME_WAIT)**
```
[Errno 98] error while attempting to bind on address ('127.0.0.1', 20001):
address already in use
```
- **Cause**: TCP ports in TIME_WAIT state from rapid start/stop cycles of Gym servers.
- **Fix**: Wait for TIME_WAIT expiry (~60s) or recreate container. No code change needed.

**Bug 9: Health check always failing — `curl -sf` on `/health` returns 404**
```
(code_gen) INFO: 127.0.0.1:... - "GET /health HTTP/1.1" 404 Not Found
[init] ERROR: Not all servers healthy after 120s
```
- **Cause**: `start_gym_uv.sh` used `curl -sf "http://localhost:${port}/health"`. The `-f` flag treats HTTP 4xx as failures. But Gym servers have NO `/health` endpoint — they use GET `/` which returns 404, and Gym's own `poll_for_status()` only checks TCP connectivity (not HTTP status code). See `Gym/nemo_gym/server_utils.py:304-312`.
- **Fix**: Changed to `curl -s --connect-timeout 2 "http://localhost:${port}/"` — checks connectivity only, matches Gym's own health check behavior.
- **Files changed**: `start_gym_uv.sh`, `plans/nemo_gym_worker/scripts/start_gym.sh`

**Bug 10: Docker push authorization failed**
```
push access denied, repository does not exist or may require authorization:
server message: insufficient_scope: authorization failed
```
- **Cause**: Registry credentials expired or `dvoff` didn't properly disable proxy before push.
- **Status**: Not a code bug. User needs to re-authenticate with Aliyun registry.

**Bug 11: Training script `--wait` flag not handled**
```
# run_moonlight_1node_blend_smoke.sh line 10 had:
bash "${SCRIPT_DIR}/start_gym_uv.sh" --wait
```
- **Cause**: `start_gym_uv.sh` only handles `--status` and `--stop`. The `--wait` flag was passed as `$1` which then got used as TIMEOUT value (line 102: `TIMEOUT="${1:-120}"`), causing `--wait` to be treated as a non-numeric timeout.
- **Fix**: Removed `--wait` from the training script. Default behavior already starts + waits for health.

### Build Result (Image ID: e79627581af2)

Build 4 succeeded. L1-L4 cached (vLLM+CUDA), L5-L13 rebuilt.

Image size: 68.5GB (content size 5.02GB delta).

**Verification results:**

| Component | Status | Details |
|-----------|--------|---------|
| verl import | OK | 0.8.0.dev, editable at `/root/myCodeLab/host/verl/` |
| tensordict | OK | 0.11.0, pyvers dependency resolved |
| transformers | OK | 4.57.3 (pin effective) |
| Gym .venv | OK | isolated at `/root/myCodeLab/host/Gym/.venv/`, nemo_gym 0.3.0rc0 |
| Gym domain deps | OK | math_verify, langdetect, openapi_schema_validator all importable in Gym .venv |
| Per-server venvs | OK | All 6: code_gen, mcqa, instruction_following, structured_outputs, workplace_assistant, math_with_judge |
| lcb_integration | OK | Symlinked to `/usr/local/lib/python3.12/dist-packages/lcb_integration` |
| Gym isolation | OK | `import nemo_gym` fails from system Python (not in /opt/venv/) |
| env.yaml | OK | Policy model placeholder in `/root/myCodeLab/host/Gym/env.yaml` |
| uv (upgraded) | OK | 0.10.9 at `~/.local/bin/uv` (base's 0.7.2 used for L1-L5) |

**.dockerignore updates:**
- Added `Gym/.venv` — exclude host's main Gym venv from build context
- Added `verifiable-instructions/` — no longer COPYed (lc_fix pulls from git)
- Removed `Gym/responses_api_models/` — needed by ng_run for policy_model config
- Kept `Gym/cache/` — created fresh in Dockerfile (avoids host cache bloat)

### Runtime Changes

**start_gym_uv.sh fixes (Bugs 4, 5, 9):**
- `--max-worker-port`: 7499 → 7999 (100 ports too few, Bug 4)
- `--num-cpus`: 256 → 32 (fd exhaustion with default ulimit, Bug 5)
- Removed `--memory` flag (let Ray auto-detect, was unnecessary)
- Health check: `curl -sf .../health` → `curl -s --connect-timeout 2 .../` (Bug 9)

**docker-compose.yml fix (Bug 5):**
- Added `nofile: soft: 65536, hard: 65536` to ulimits section

**Training script fix (Bug 11):**
- `run_moonlight_1node_blend_smoke.sh` line 10: removed `--wait` (not handled by script)

**Venv extraction (Bug 6):**
- Extracted Docker image's per-server venvs to local filesystem via `docker cp`
- Ensures mount overlay sees correct venvs (matching Ray versions, correct deps)

### Testing Status — PASSED

**Image verification**: All components verified inside Docker image (see table above).

**Gym server startup**: Fresh container → `start_gym_uv.sh` → all 7/7 servers healthy.
```
All 7 / 7 servers ready! Polling every 60s
[init] All 6 Gym servers healthy.
```

**Scoring test suite**: 30/30 samples scored correctly across all 6 domains (4.1s total).
```
nemogym_math              5/5 (1.00)
nemogym_mcqa              5/5 (1.00)
nemogym_if                5/5 (1.00)
nemogym_code              5/5 (1.00)
nemogym_structured        5/5 (1.00)
nemogym_workplace         5/5 (1.00)
```
Test: `python plans/nemo_gym_worker/scoring/tests/test_scoring.py`

**Operations**:
```bash
# Start servers
docker exec vrl bash -c 'cd /root/myCodeLab/host/verl && bash my_scripts/start_gym_uv.sh'

# Check status
docker exec vrl bash -c 'cd /root/myCodeLab/host/verl && bash my_scripts/start_gym_uv.sh --status'

# Run training
docker exec vrl bash -c 'cd /root/myCodeLab/host/verl && bash my_scripts/run_moonlight_1node_blend_smoke.sh'
```

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
