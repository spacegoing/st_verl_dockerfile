# README — verl Moonlight-16B Training Docker Image (nemo:26.02)

Image: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`
Cluster: b31 (10.12.11.5) + b32 (10.12.11.6), 8× B300 SXM6 each

---

## Dockerfile Layer-by-Layer Rationale

**File**: `Dockerfile.ncr.26.02.mydev`
**Base**: `nvcr.io/nvidia/nemo:26.02` (~107 layers)

The layer order is designed so that the slowest, most stable layers are built
first (maximizing Docker cache hits), and the fastest, most frequently changed
layers are last. This means editing verl or mjnemogym source triggers only a
~30s rebuild (layers 10-12), not a 30-minute vLLM recompile.

**Build constraint**: `DOCKER_BUILDKIT=0` required. The base image has ~107
layers; the legacy builder's 127-layer limit demands consolidation. BuildKit
is not used because its layer counting differs and we hit provenance flag errors.

### Layer 1 — vLLM prep

```dockerfile
RUN pip uninstall -y vllm && pip install --no-cache-dir setuptools_scm
```

**Why**: 26.02 ships vLLM 0.14.2 as a pip package (no `/opt/vllm/` dir anymore).
verl requires exactly 0.12.0 because 0.13+ changed the rollout worker API.
Uninstalling first prevents import conflicts. `setuptools_scm` is needed because
vLLM's build system requires it, and the `.git/` directory is stripped from the
COPY'd source (to save 1.6GB build context).

### Layer 2 — COPY vLLM source

```dockerfile
COPY vllm /opt/vllm
```

**Why**: Placed before the build layer so that source changes invalidate only
layers 2-3, not the entire image. The source in `vllm/` must be pre-cleaned:
delete `.git/`, `.deps/`, `tests/`, `docs/`, `benchmarks/`, `examples/`, and
all `*.so` files. This reduces COPY context from 3.8GB to ~26MB.

### Layer 3 — vLLM CUDA build (~30 min)

```dockerfile
RUN cd /opt/vllm && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0 MAX_JOBS=64 NVCC_THREADS=1 \
    pip install --no-deps --no-build-isolation --no-cache-dir -e .
```

**Why this is layer 3 (not later)**: This is the slowest step. Placing it early
means any change to verl/mjnemogym/configs does NOT trigger a 30-minute rebuild.

**`--no-build-isolation`**: Without this, pip creates an isolated venv and
downloads its own torch (CUDA 12.6), which conflicts with the base image's
torch 2.10.0a0 (CUDA 13.0). All CUDA extensions must link against the base torch.

**`--no-deps`**: Protects all base image packages from being modified. See
`requirements_2602.md` A10 for the full protected list.

**`SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0`**: Without `.git/`, setuptools_scm
cannot detect the version. This fakes it.

**`MAX_JOBS=64 NVCC_THREADS=1`**: Maximizes parallel C++ compilation while
keeping each NVCC invocation single-threaded (NVCC is memory-hungry; parallel
NVCC threads cause OOM on build machines).

**`-e .` (editable)**: The `.pth` file records `/opt/vllm` as the import path.
Since `/opt/vllm` is NOT under the docker-compose mount, vLLM always resolves
from the image layer (immutable). This is intentional — vLLM is a build artifact,
not a live-edited source.

### Layer 4 — CUDA extension packages

```dockerfile
RUN pip install --no-cache-dir --no-deps --no-build-isolation grouped_gemm && \
    pip install --no-cache-dir --no-deps --no-build-isolation causal_conv1d && \
    pip install --no-cache-dir --no-deps --no-build-isolation mamba_ssm
```

**Why**: These were in nemo:25.11.01 but removed in 26.02. They're imported by
megatron-core (MoE expert GEMM, SSM architecture). Each requires NVCC compilation,
hence `--no-build-isolation` (same torch ABI reason as vLLM). Combined into one
RUN to save layers. Placed after vLLM because they're faster (~5 min total) and
change even less frequently.

### Layer 5 — System packages + pure-Python pip deps

```dockerfile
RUN apt-get update && apt-get install -y pdsh tmux htop vim && \
    rm -rf /var/lib/apt/lists/* && \
    git config --global --add safe.directory '*' && \
    pip install --no-cache-dir --no-deps \
        wandb gpustat codetiming tensordict mathruler pylatexenc torchdata \
        hydra-core bitsandbytes orjson \
        transformers==4.57.3 \
        model_hosting_container_standards anthropic \
        pyvers math_verify latex2sympy2_extended openapi_schema_validator && \
    wandb login df3cecbfc0874c8a352c40820becf4a15575614e
```

**Why all in one RUN**: Layer budget. Combining apt + pip + wandb login into one
layer saves 2 layers. These are all pure-Python or pre-built wheels — no NVCC
needed, so `--no-build-isolation` is not required.

**`--no-deps` for every pip install**: The golden rule of this Dockerfile. Without
it, installing `wandb` would pull `numpy>=2.0` which would overwrite the base
image's `numpy==1.26.4`, breaking every CUDA extension's C ABI.

**`transformers==4.57.3`**: Overrides the base's 4.56.0. verl and the Moonlight
model config require features from 4.57+.

**`git config --global --add safe.directory '*'`**: Without this, git operations
inside mounted volumes fail with "dubious ownership" errors.

**`wandb login`**: Baked in so training can write wandb logs without interactive
login. Runs offline (`WANDB_MODE=offline` in compose).

**apt packages**: `pdsh` for multi-node commands, `tmux`/`htop`/`vim` for dev.

### Layer 6 — mbridge

```dockerfile
COPY mbridge /tmp/mbridge
RUN cd /tmp/mbridge && pip3 install --no-cache-dir --no-deps . && rm -rf /tmp/mbridge
```

**Why**: mbridge (v0.15.1) is a pure-Python bridge between verl and megatron-core
with zero declared dependencies. Installed as a regular (non-editable) package
because it rarely changes. COPY to `/tmp/` + install + delete keeps the image
clean. Separate from layer 5 because it's a local source install, not a PyPI
download.

### Layer 7 — Static config files

```dockerfile
COPY o200k_base.tiktoken to_append.sh .tmux.conf /tmp/docker_context/
COPY nltk_data /usr/local/share/nltk_data
```

**`o200k_base.tiktoken`**: vLLM 0.12 uses OpenAI's tiktoken encoding. Without
this file cached locally, vLLM would try to download it at runtime (no internet).

**`nltk_data/`**: NLTK tokenization data used by mjnemogym's math evaluation.
NLTK's default download path is `/usr/local/share/nltk_data`.

**`to_append.sh`**: Shell aliases (`tnew`, `tat`, `tkl` for tmux; `gd` to cd to
verl). Appended to `.bashrc` in layer 9.

**`.tmux.conf`**: tmux configuration. Moved to `/root/` in layer 9.

### Layer 8 — Environment variables

```dockerfile
ENV NLTK_DATA=/usr/local/share/nltk_data \
    TIKTOKEN_ENCODINGS_BASE=/root/tiktoken_cache
```

**Why**: Tells NLTK and tiktoken where to find their data files (COPY'd in
layers 7 and 9). These are build-time ENV, not runtime — they're baked into the
image so they're always available regardless of how the container is started.

### Layer 9 — Workspace setup

```dockerfile
RUN mkdir -p /root/myCodeLab/host /root/myCodeLab/public /root/tiktoken_cache && \
    mv /tmp/docker_context/o200k_base.tiktoken /root/tiktoken_cache/ && \
    echo "" >> /root/.bashrc && \
    cat /tmp/docker_context/to_append.sh >> /root/.bashrc && \
    mv /tmp/docker_context/.tmux.conf /root/ && \
    rm -rf /tmp/docker_context && \
    ln -s /usr/local/lib/python3.12/dist-packages /root/myCodeLab/dist-packages
```

**`mkdir -p /root/myCodeLab/host`**: Creates the directory that docker-compose
will mount over. In the image, it's empty. At runtime, the mount overlays it
with the host project directory. **No symlinks, no host path assumptions**.

**`dist-packages` symlink**: Convenience shortcut to quickly inspect installed
packages (`ls /root/myCodeLab/dist-packages/`).

### Layers 10-11 — COPY codebases (change most often)

```dockerfile
COPY verl /root/myCodeLab/host/verl/
COPY MJ_NEMO_GYM /root/myCodeLab/host/MJ_NEMO_GYM/
```

**Why last**: These directories change with every code edit. Placing them last
means a code change only invalidates layers 10-14 (~30s rebuild), not the entire
image.

**Critical — directory names must match host**: The COPY destination names MUST
match the directory names on the host filesystem (`verl/` and `MJ_NEMO_GYM/`).
This is because:
1. Layer 12 does `pip install -e .` which creates `.pth` files in `dist-packages/`
2. The `.pth` files record the absolute path (e.g., `/root/myCodeLab/host/MJ_NEMO_GYM/`)
3. At runtime, docker-compose mounts the host dir over `/root/myCodeLab/host/`
4. The `.pth` files survive (they're in `dist-packages/`, not under the mount)
5. The `.pth` paths must resolve to directories that exist in the mount

If we used `COPY MJ_NEMO_GYM /root/myCodeLab/host/mjnemogym/`, the `.pth` would
record `mjnemogym/` but the mount would provide `MJ_NEMO_GYM/` → import failure.
This was the root cause of Bug #4 in v1.

### Layer 12 — Editable install

```dockerfile
RUN cd /root/myCodeLab/host/verl && pip3 install --no-deps -e . && \
    cd /root/myCodeLab/host/MJ_NEMO_GYM && pip3 install --no-cache-dir --no-deps -e .
```

**Why editable (`-e`)**: At runtime, the docker-compose mount replaces the COPY'd
source with live host source. The `.pth` files (in `dist-packages/`) point to
paths under `/root/myCodeLab/host/` — which now resolve through the mount to
live code. Result: editing code on the host is immediately reflected inside the
container with zero rebuild.

**Why `--no-deps`**: Same reason everywhere. verl's `setup.py` declares deps like
`numpy`, `transformers`, etc. Installing with deps would corrupt the base image.

### Layers 13-14 — Final

```dockerfile
WORKDIR /root/myCodeLab
CMD ["/bin/bash"]
```

**WORKDIR**: Default working directory when entering the container.
**CMD**: Interactive shell. Overridden by docker-compose `command:` for ray start.

---

## Docker Compose Section-by-Section Rationale

**File**: `docker-compose.yml`

### Anchor: `x-common`

Shared config for both head and worker services. Changes apply to both nodes.

### `image`

```yaml
image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
```

Aliyun Container Registry. Both nodes must pull the same image to guarantee
identical environments (Bug #16: different image IDs caused confusion).

### `entrypoint` / `stdin_open` / `tty`

```yaml
entrypoint: /bin/bash
stdin_open: true
tty: true
```

Override the image's CMD to allow `docker exec -it vrl bash` for interactive
debugging. Without `stdin_open` + `tty`, the container would exit immediately
after ray starts (only `tail -f /dev/null` keeps it alive).

### `network_mode: host`

```yaml
network_mode: host
```

**Why**: Ray and NCCL need direct access to the host network stack. Ray head
binds to port 6379 on the host IP; NCCL communicates over RDMA/RoCE using the
host's InfiniBand devices. Bridge networking would add latency, break RDMA, and
require port mapping for every NCCL connection.

### `ipc: host`

```yaml
ipc: host
```

**Why**: PyTorch uses shared memory (`/dev/shm`) for inter-process GPU tensor
transfers. Docker's default 64MB shared memory is far too small for 275GB-per-GPU
tensors. `ipc: host` gives the container the host's full shared memory space.

### `privileged: true`

```yaml
privileged: true
```

**Why**: Required for RDMA/InfiniBand access. Without it, the container cannot
see `/dev/infiniband/` devices (Bug #12). This gives NCCL access to the mlx5
RoCE NICs for GPU-direct RDMA. Downside: full device access. Acceptable for a
dedicated training cluster.

### `ulimits`

```yaml
ulimits:
  memlock: { soft: -1, hard: -1 }
  stack: { soft: 67108864, hard: 67108864 }
```

**`memlock: -1`**: Unlimited locked memory. RDMA (InfiniBand verbs) requires
pinning GPU memory buffers. The default 64KB limit would cause `ibv_reg_mr`
failures on every NCCL collective.

**`stack: 67108864`** (64MB): Deep call stacks in megatron-core's pipeline
parallelism and vLLM's async engine. Default 8MB can cause stack overflows
during model initialization.

### `volumes`

```yaml
volumes:
  - /mnt/public:/public
  - /mnt/public/lichang93/st_verl_dockerfile:/root/myCodeLab/host
```

**`/mnt/public:/public`**: The full shared filesystem (quarkfs, 1PB). Needed for:
- Cross-node checkpoint saving (`/public/lichang93/...`)
- Model/dataset access if not using the project mount
- Any shared data between nodes

**`st_verl_dockerfile:/root/myCodeLab/host`**: The project root. This is the
mount that makes editable installs work at runtime (see Layers 10-12 above).
Contains: `verl/`, `MJ_NEMO_GYM/`, `downloads/` (symlink to stCodeLab/downloads),
training scripts, configs, wandb logs, checkpoints.

**No symlinks in the image**: v1 used `ln -s /public/lichang93/stCodeLab` inside
the image, which broke on clusters where that path didn't exist (Bug #1, #3, #17).
v2 uses a plain empty dir + compose mount. The only symlink is on the host:
`downloads -> /mnt/public/lichang93/stCodeLab/downloads` (one-time setup for
30GB model data).

### Environment Variables

Grouped by subsystem:

#### Core

```yaml
CUDA_DEVICE_MAX_CONNECTIONS: "1"
```
Limits CUDA streams per device. Required by megatron-core for correct pipeline
parallelism scheduling — ensures operations on the same GPU are serialized
through a single stream, preventing race conditions in pipeline stage transitions.

#### vLLM

```yaml
VLLM_USE_V1: "1"
VLLM_ATTENTION_BACKEND: "CUTLASS_MLA"
```

**`VLLM_USE_V1`**: Enables vLLM v1 engine (async, better memory management).

**`VLLM_ATTENTION_BACKEND=CUTLASS_MLA`**: **B300 hardware workaround** (Bug #15).
vLLM's `is_device_capability(100)` does an exact match, but B300 reports sm_103.
This means vLLM doesn't detect it as Blackwell, and falls back to FlashMLA Dense
(Hopper-only path) which fails with `ValueError: No common block size for 16`.
Forcing CUTLASS_MLA selects a block_size=128 codepath that works on sm_103.
Requires cuDNN >= 9.11 for MLA fused attention on sm >= 100; image has 9.18.

#### NCCL Networking

```yaml
NCCL_SOCKET_IFNAME: bond0
NCCL_IB_HCA: "mlx5_10:1,mlx5_11:1,...,mlx5_17:1"
NCCL_IB_GID_INDEX: "3"
NCCL_IB_TC: "138"
NCCL_IB_QPS_PER_CONNECTION: "8"
NCCL_MIN_NCHANNELS: "32"
NCCL_RUNTIME_CONNECT: "0"
NCCL_NVLS_ENABLE: "1"
```

**`NCCL_SOCKET_IFNAME=bond0`**: Tells NCCL to use the `bond0` interface for
out-of-band control messages (not data). Warning: `bond0` prefix-matches
`bond0.200` (VLAN), which would route to the wrong subnet. In privileged mode,
NCCL uses IB for data and only falls back to socket for bootstrap — so this
works because IB is available.

**`NCCL_IB_HCA`**: Explicitly lists the 8 RoCE NICs (mlx5_10 through mlx5_17),
one per GPU. Each GPU has PXB (PCIe bridge) affinity to its NIC:
GPU0↔mlx5_10, GPU1↔mlx5_11, ..., GPU7↔mlx5_17. The `:1` suffix selects
port 1 on each HCA.

**`NCCL_IB_GID_INDEX=3`**: **Critical for B300 RoCE** (Bug #14). The IB GID
table has 4 entries:
- GID 0: `fe80::...` (link-local IPv6, not routable cross-node)
- GID 1: `fe80::...` (link-local)
- GID 2: `::ffff:100.86.0.x` (IPv4-mapped, RoCEv2)
- GID 3: `::ffff:100.86.0.x` (IPv4-mapped, RoCEv2, routable)

Index 0 (default) gives `fe80::` which is link-local only → NCCL gets
`status=12 vendor err 129` on cross-node send. Index 3 gives routable RoCEv2.

**`NCCL_IB_TC=138`**: Traffic class for RDMA. Matches the cluster's QoS
configuration for GPU training traffic. Must also match NVSHMEM's
`NVSHMEM_IB_TRAFFIC_CLASS=138`.

**`NCCL_IB_QPS_PER_CONNECTION=8`**: 8 Queue Pairs per connection. With 8 GPUs
per node, this gives 64 QPs per node pair — saturates the 400Gb/s per-NIC
bandwidth.

**`NCCL_MIN_NCHANNELS=32`**: Minimum NCCL channels. More channels = more
parallelism in collective operations. 32 is optimal for 8-GPU nodes with
400Gb/s RoCE.

**`NCCL_RUNTIME_CONNECT=0`**: Establish all NCCL connections at init time, not
lazily. Prevents timeout errors during the first collective operation after a
long idle period (e.g., after model loading).

**`NCCL_NVLS_ENABLE=1`**: Enables NVLink SHARP (NVSwitch-based in-network
reductions). B300 SXM6 has NVSwitch — this accelerates allreduce by doing
partial reductions in the switch fabric.

#### NVSHMEM / DeepEP

```yaml
NVSHMEM_IBGDA_NIC_HANDLER: gpu
NVSHMEM_IB_TRAFFIC_CLASS: "138"
NVSHMEM_IB_ENABLE_IBGDA: "1"
```

**`NVSHMEM_IBGDA_NIC_HANDLER=gpu`**: GPU-initiated RDMA. DeepEP (deep expert
parallelism) uses NVSHMEM for GPU-to-GPU communication in MoE layers. The `gpu`
handler means the GPU directly initiates RDMA writes without CPU involvement —
critical for low-latency expert dispatch.

**`NVSHMEM_IB_TRAFFIC_CLASS=138`**: Must match NCCL's TC for consistent QoS.

**`NVSHMEM_IB_ENABLE_IBGDA=1`**: Enables InfiniBand GPU Direct Async. Required
for DeepEP's zero-copy expert-to-expert tensor transfers.

#### Transformer Engine

```yaml
NVTE_FUSED_ATTN: "1"
```

Enables fused attention kernels in NVIDIA Transformer Engine. Moonlight-16B uses
MLA (Multi-head Latent Attention). On B300 (sm_103), this requires cuDNN >= 9.11
for the MLA fused attention path. The base image ships cuDNN 9.18, so this works.

#### Ray

```yaml
RAY_USAGE_STATS_ENABLED: "0"
RAY_ENABLE_OPENTELEMETRY: "0"
TORCH_NCCL_ASYNC_ERROR_HANDLING: "1"
```

**Disabled telemetry**: No internet at runtime. Without these, Ray prints
warnings on every call trying to phone home.

**`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`**: PyTorch detects hung NCCL collectives
and raises errors instead of deadlocking. Essential for debugging multi-node
training hangs.

#### wandb

```yaml
WANDB_MODE: offline
```

No internet at runtime. Wandb writes logs locally to
`/root/myCodeLab/host/verl/wandb_my_dirs/` (persisted on shared fs via mount).
Can be synced later with `wandb sync`.

### `deploy.resources`

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Reserves all 8 B300 GPUs. `count: all` rather than `count: 8` so the compose
file works on machines with different GPU counts without modification.

### `restart: unless-stopped`

Training crashes should not silently restart (would corrupt checkpoints / wandb
state). `unless-stopped` means: restart on Docker daemon restart (e.g., after
`dvon`/`dvoff` which restarts the daemon), but don't restart if the container
was explicitly stopped.

### Services: `head` and `worker`

```yaml
services:
  head:
    container_name: vrl
    hostname: b31
    extra_hosts: ["b31:10.12.11.5", "b32:10.12.11.6"]
    command:
      - -c
      - >-
        ray start --head --port=6379 --dashboard-port=8265
        --num-gpus=8 --num-cpus=64
        --node-ip-address=10.12.11.5
        && tail -f /dev/null
```

**`container_name: vrl`**: Same name on both nodes for consistent `docker exec`
commands. Since `network_mode: host`, there's no container-to-container collision.

**`hostname`**: Sets the container hostname for Ray node identification.

**`extra_hosts`**: Injects `/etc/hosts` entries so ray/NCCL can resolve node
names. Required because `network_mode: host` means the container sees the host's
`/etc/hosts`, but we add these entries explicitly for portability.

**`command` format**: Uses YAML list (`["-c", ">- ..."]`) instead of the `>`
folded scalar. Bug #13: YAML `>` folded scalar splits the command at indentation
boundaries, breaking the `ray start` command. The list format passes the entire
string as a single `-c` argument to `/bin/bash`.

**`ray start --head`**: Starts Ray head node on b31. Port 6379 is Ray's default
GCS port. Dashboard on 8265 for monitoring.

**`--num-gpus=8 --num-cpus=64`**: Advertises resources to Ray. verl's resource
scheduler uses these to place rollout workers (vLLM) and training workers
(megatron) across GPUs.

**`--node-ip-address`**: Explicit IP avoids Ray auto-detecting the wrong
interface (e.g., docker bridge, VLAN interface).

**`tail -f /dev/null`**: Keeps the container alive after ray starts. Without it,
the shell exits and the container stops.

**Worker**: Same as head but `ray start --address=10.12.11.5:6379` to join the
head's cluster.

---

## Hardware Requirements

| Resource | Specification | Why |
|----------|--------------|-----|
| GPU | 8× NVIDIA B300 SXM6 per node (sm_103, 275GB HBM) | Moonlight-16B uses ~209GB/GPU with MoE experts |
| GPU interconnect | NVSwitch (NVLink) | Intra-node allreduce, NVLS |
| Network | 8× 400Gb/s RoCE NICs per node (mlx5_10-17) | Per-GPU NIC affinity (PXB topology) for RDMA |
| InfiniBand | mlx5_z0-z3 (IB devices) + mlx5_10-17 (RoCE) | Cross-node NCCL collectives + DeepEP |
| cuDNN | >= 9.11 (image has 9.18) | MLA fused attention on sm >= 100 |
| Shared filesystem | /mnt/public (quarkfs, mounted on all nodes) | Model weights, datasets, checkpoints, source code |
| RAM | ~512GB per node | Model loading, Ray overhead, data preprocessing |

---

## Software Requirements

| Component | Version | Location |
|-----------|---------|----------|
| CUDA | 13.0 | Base image |
| torch | 2.10.0a0+nv25.11 | Base image (DO NOT MODIFY) |
| megatron-core | 0.16.0 | `/opt/Megatron-Bridge/3rdparty/Megatron-LM/` |
| transformer_engine | 2.12.0 | Base image |
| vLLM | 0.12.0 | `/opt/vllm/` (built from source, editable) |
| deep_ep | 1.2.1 | Base image |
| flash_attn | 2.7.4.post1+25.11 | Base image |
| Ray | 2.54.0 | Base image |
| Python | 3.12 | Base image |

---

## Quick Reference

```bash
# Build (on b31):
dvon
DOCKER_BUILDKIT=0 docker build -f Dockerfile.ncr.26.02.mydev --network host \
  -t registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev .
dvoff
docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev

# Pull (on b32):
docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev

# Start cluster:
# b31: docker compose up -d head
# b32: docker compose -f /mnt/public/lichang93/st_verl_dockerfile/docker-compose.yml up -d worker

# Launch training:
docker exec -it vrl bash
cd /root/myCodeLab/host/verl/my_scripts
bash run_moonlight_2node_test.sh
```

See `requirements_2602.md` for the exhaustive dependency list and `dev_manual.md`
for the full bugs & fixes log.
