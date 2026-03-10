# verl Image Test Plan: Moonlight-16B GRPO Training on 2-Node B300 Cluster

**Image**: `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev`
**Base**: `nvcr.io/nvidia/nemo:26.02`
**Target**: Moonlight-16B (MoE, 16B total / 3B active) GRPO training with Megatron backend + vLLM rollout + DeepEP

---

## Cluster Topology

| Node | Hostname | IP | GPUs | VRAM | NV-Link | NICs (RDMA) |
|------|----------|----|------|------|---------|-------------|
| Head | ecs-cb37e8df-d4d6 | 10.12.11.5 | 8× B300 SXM6 AC | 275GB×8 | NV18 | mlx5_10–mlx5_17 (400Gb/s each) |
| Worker | ecs-cb37e8df-6762 | 10.12.11.6 | 8× B300 SXM6 AC | 275GB×8 | NV18 | mlx5_10–mlx5_17 (400Gb/s each) |

**Shared filesystem**: `/mnt/public` (quarkfs, 1PB, mounted on both nodes)
**GPU-NIC affinity** (PXB = same PCIe switch):
- GPU0↔mlx5_10, GPU1↔mlx5_11, GPU2↔mlx5_12, GPU3↔mlx5_13
- GPU4↔mlx5_14, GPU5↔mlx5_15, GPU6↔mlx5_16, GPU7↔mlx5_17

---

## Phase 0: Prerequisites (Before Any Container)

### 0.1 Missing Data — Must Download First

| Item | HuggingFace ID | Target Path | Size Est. |
|------|---------------|-------------|-----------|
| Moonlight-16B | `moonshotai/Moonlight-16B-A3B` | `/mnt/public/lichang93/downloads/models/Moonlight16B` | ~32GB |
| DAPO-Math-17k | `BytedTsinghua-SIA/DAPO-Math-17k` | `/mnt/public/lichang93/downloads/datasets/dapo_data/dapo-math-17k.parquet` | ~50MB |
| AIME-2024 | `BytedTsinghua-SIA/AIME-2024` | `/mnt/public/lichang93/downloads/datasets/dapo_data/aime-2024.parquet` | ~1MB |

**Download commands** (run on either node, shared fs):
```bash
mkdir -p /mnt/public/lichang93/downloads/{models,datasets/dapo_data}

# Model (large, use git-lfs or huggingface-cli)
pip install huggingface_hub[cli]
huggingface-cli download moonshotai/Moonlight-16B-A3B \
  --local-dir /mnt/public/lichang93/downloads/models/Moonlight16B

# Datasets
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k \
  --local-dir /mnt/public/lichang93/downloads/datasets/dapo_data/dapo-math-17k \
  --include "*.parquet"
huggingface-cli download BytedTsinghua-SIA/AIME-2024 \
  --local-dir /mnt/public/lichang93/downloads/datasets/dapo_data/aime-2024 \
  --include "*.parquet"
# Then move/symlink .parquet files to expected paths
```

**CRITICAL**: After downloading Moonlight-16B, edit its `config.json`:
1. Remove `quantization_config` block (if present)
2. Set `num_nextn_predict_layers: 0` (disable MTP, not supported)

### 0.2 /etc/hosts on Both Nodes

```bash
# Run on BOTH nodes
cat >> /etc/hosts << 'EOF'
10.12.11.5 b31
10.12.11.6 b32
EOF
```

### 0.3 SSH Keys

Verify passwordless SSH between nodes (already working based on tests).

### 0.4 Docker Image on b32

```bash
# Option A: Push to registry, pull on b32
docker push registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
ssh b32 "docker pull registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev"

# Option B: Save/load via shared fs (faster if registry is slow)
docker save registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev \
  | pigz > /mnt/public/lichang93/myverl_ncr2602.tar.gz
ssh b32 "docker load < /mnt/public/lichang93/myverl_ncr2602.tar.gz"
```

---

## Phase 1: Single-Container Smoke Tests

### 1.1 Launch Container on Head Node

```bash
docker run --entrypoint /bin/bash -itd \
  --name vrl \
  --network host \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add=IPC_LOCK \
  -v /mnt/public:/public \
  registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev

docker exec -it vrl bash
```

### 1.2 Import Checks

```bash
# Inside container
python3 -c "
import torch; print('torch', torch.__version__, torch.cuda.device_count(), 'GPUs')
import vllm; print('vllm', vllm.__version__)
import verl; print('verl OK')
import megatron.core; print('megatron-core OK')
import transformer_engine; print('TE', transformer_engine.__version__)
import mbridge; print('mbridge OK')
import deep_ep; print('deep_ep OK')
import flash_attn; print('flash_attn', flash_attn.__version__)
import flashinfer; print('flashinfer', flashinfer.__version__)
import hydra; print('hydra OK')
import wandb; print('wandb OK')
import ray; print('ray', ray.__version__)
import mamba_ssm; print('mamba_ssm OK')
import causal_conv1d; print('causal_conv1d OK')
import grouped_gemm; print('grouped_gemm OK')
import mjnemogym; print('mjnemogym OK')
print('ALL IMPORTS OK')
"
```

### 1.3 GPU / NCCL Sanity

```bash
# Inside container
python3 -c "
import torch
import torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group('nccl', rank=0, world_size=1)
t = torch.ones(1024, device='cuda:0')
print('NCCL init OK, tensor sum:', t.sum().item())
dist.destroy_process_group()
"
```

### 1.4 vLLM Inference Smoke Test (Single GPU)

```bash
python3 -c "
from vllm import LLM, SamplingParams
import os
os.environ['VLLM_USE_V1'] = '1'
model_path = os.path.expanduser('~/myCodeLab/host/downloads/models/Moonlight16B')
llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True,
           gpu_memory_utilization=0.5, max_model_len=1024)
out = llm.generate(['What is 2+2?'], SamplingParams(max_tokens=64, temperature=0.7))
print(out[0].outputs[0].text)
print('vLLM inference OK')
"
```

### 1.5 Verify Data Paths

```bash
python3 -c "
import pandas as pd
train = pd.read_parquet('/root/myCodeLab/host/downloads/datasets/dapo_data/dapo-math-17k.parquet')
test = pd.read_parquet('/root/myCodeLab/host/downloads/datasets/dapo_data/aime-2024.parquet')
print(f'Train: {len(train)} rows, columns: {list(train.columns)}')
print(f'Test: {len(test)} rows, columns: {list(test.columns)}')
assert 'prompt' in train.columns, 'Missing prompt column in train'
assert 'prompt' in test.columns, 'Missing prompt column in test'
print('Data OK')
"
```

---

## Phase 2: Multi-Node Ray Cluster

### 2.1 Start Ray Head (on b31 container)

```bash
ray start --head --port=6379 --dashboard-port=8265 \
  --num-gpus=8 --num-cpus=64 \
  --node-ip-address=10.12.11.5
```

### 2.2 Start Ray Worker (on b32 container)

```bash
# SSH into b32, docker exec into container
ray start --address=10.12.11.5:6379 \
  --num-gpus=8 --num-cpus=64 \
  --node-ip-address=10.12.11.6
```

### 2.3 Verify Ray Cluster

```bash
# On head node
ray status
# Should show: 2 nodes, 16 GPUs, 128 CPUs
python3 -c "import ray; ray.init('auto'); print(ray.cluster_resources()); ray.shutdown()"
```

### 2.4 Multi-Node NCCL Test

```bash
# Quick all-reduce test across 16 GPUs
python3 -c "
import ray
ray.init('auto')

@ray.remote(num_gpus=1)
def nccl_test():
    import torch, os, socket
    return f'{socket.gethostname()} GPU:{os.environ.get(\"CUDA_VISIBLE_DEVICES\",\"?\")} torch.cuda.is_available={torch.cuda.is_available()}'

results = ray.get([nccl_test.remote() for _ in range(16)])
for r in results:
    print(r)
ray.shutdown()
"
```

---

## Phase 3: Training Parameter Adaptation (64-node → 2-node)

The original script targets 64 nodes (512 GPUs). With 2 nodes (16 GPUs), key parameters need scaling:

### 3.1 Parallelism Configuration

| Parameter | 64-node value | 2-node value | Rationale |
|-----------|---------------|--------------|-----------|
| `NNODES` | 64 | 2 | Physical constraint |
| `train_pp` | 2 | 2 | Keep PP=2 for memory |
| `vpp` | 2 | 2 | Keep VPP=2 |
| `train_tp` | 1 | 1 | Keep TP=1 |
| `EP` | 8 | 8 | 16 GPUs / (PP=2 × TP=1) = 8 experts per EP group — exactly matches |
| `gen_tp` | 1 | 1 | Keep |
| `CP` | 1 | 1 | Keep |

### 3.2 Batch Size Scaling

| Parameter | 64-node value | 2-node value | Rationale |
|-----------|---------------|--------------|-----------|
| `train_prompt_bsz` | 128 | 8 | Scale by ~nodes ratio, must be ≥ n_resp (or divisible) |
| `n_resp_per_prompt` | 16 | 8 | Reduce to fit in 16 GPUs memory |
| `train_prompt_mini_bsz` | 128 | 8 | = train_prompt_bsz for simplicity |
| `ppo_epochs` | 4 | 1 | Reduce for faster iteration during testing |

### 3.3 Sequence Length (For Initial Testing)

| Parameter | 64-node value | Test value | Rationale |
|-----------|---------------|------------|-----------|
| `max_prompt_length` | 8000 | 2048 | Reduce for faster testing |
| `max_response_length` | 40960 | 4096 | Reduce significantly for testing |

### 3.4 Create Test Launch Script

Create `run_moonlight_2node_test.sh` — a modified version of the 64-node script with the above parameters. See Phase 5 for the script.

---

## Phase 4: Debugging Workflow

### 4.1 Iterative Development Cycle

The container mounts `/mnt/public:/public`, and `/public/lichang93/stCodeLab` is symlinked to `/root/myCodeLab/host`. This means:

1. **Edit verl code** on host at `/mnt/public/lichang93/stCodeLab/verl/` (or inside container at `/root/myCodeLab/host/verl/`)
2. **No rebuild needed** — verl is editable-installed (`pip install -e .`)
3. **Restart Ray job** to pick up changes

### 4.2 Common Failure Modes & Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `NCCL timeout` | Wrong NIC config or firewall | Set `NCCL_SOCKET_IFNAME=bond0`, check `NCCL_IB_HCA` |
| `CUDA OOM` during actor training | `actor_ppo_max_token_len` too high | Reduce `max_response_length` or increase `train_pp` |
| `CUDA OOM` during vLLM rollout | `gpu_memory_utilization` too high | Reduce from 0.7 to 0.5 |
| `Ray worker died` | OOM or NCCL hang | Check `ray logs`, add `NCCL_DEBUG=INFO` |
| `DeepEP NVSHMEM error` | Wrong NVSHMEM env vars | Verify `NVSHMEM_IB_ENABLE_IBGDA=1`, check NIC names |
| `mbridge error` | Version mismatch with megatron-core | Check mbridge<>megatron-core API compatibility |
| `ModuleNotFoundError` | Missing dep in container | `pip install --no-deps <pkg>` inside container, then add to Dockerfile |
| `protobuf error` | 4.x→6.x breaking change | Pin with `pip install protobuf==4.25.8 --no-deps` |
| `vLLM model load error` | MTP not disabled in config.json | Remove `quantization_config`, set `num_nextn_predict_layers=0` |

### 4.3 Logging & Monitoring

```bash
# wandb (offline mode) — logs saved to:
ls /root/myCodeLab/host/verl/wandb_my_dirs/

# Ray dashboard (if accessible):
# http://10.12.11.5:8265

# GPU monitoring:
watch -n1 gpustat -cup

# Training log:
tail -f /root/myCodeLab/host/verl/logs/run_*.log
```

### 4.4 NCCL Debug (If Communication Issues)

Add to `my_deepep_env.yaml` temporarily:
```yaml
NCCL_DEBUG: "INFO"
NCCL_DEBUG_SUBSYS: "INIT,GRAPH,NET"
```

### 4.5 NIC Configuration for This Cluster

The env yaml has generic NVSHMEM vars. For this B300 cluster, the NICs are `mlx5_10`–`mlx5_17` (NOT `SLOT0`–`SLOT7`). Update if DeepEP requires explicit NIC names:
```yaml
NVSHMEM_BOOTSTRAP_IB_DEV: "mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
NVSHMEM_HCA_LIST: "mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
NCCL_IB_HCA: "mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_14:1,mlx5_15:1,mlx5_16:1,mlx5_17:1"
```

---

## Phase 5: Training Runs

### 5.1 Minimal Smoke Run (1 step, tiny sequences)

Goal: Verify the full pipeline works end-to-end.

```bash
# Inside head container, cd /root/myCodeLab/host/verl
NNODES=2 EP=8 \
  OFFLOAD_OPTIM=True OFFLOAD_FRACTION=1.0 \
  bash my_scripts/run_moonlight_2node_test.sh
```

Expected: completes 1 training step + 1 validation without crash.

### 5.2 Short Training Run (10 steps)

After smoke test passes, run 10 steps with production sequence lengths to verify memory fits:
- `max_prompt_length=8000`, `max_response_length=8192` (half of production)
- `train_prompt_bsz=8`, `n_resp_per_prompt=8`

### 5.3 MFU Measurement

Monitor via wandb logs or console output:
- `train/tflops_per_gpu` — target depends on B300 peak FLOPS
- B300 SXM6: ~2.25 PFLOPS FP8, ~1.13 PFLOPS FP16/BF16
- Good MFU for MoE training with PP: 30-45% of peak

### 5.4 Full Production Run (If Memory Allows)

Scale back toward production params:
- `max_prompt_length=8000`, `max_response_length=40960`
- `train_prompt_bsz=8` (limited by 16 GPUs)
- `n_resp_per_prompt=16`

---

## Phase 6: Image Finalization

Once training works:

1. **Record any runtime pip installs** needed during debugging → add to Dockerfile
2. **Rebuild image** with fixes
3. **Tag final image**: `myverl:ncr2602_vllm012_verified`
4. **Distribute to all nodes**

---

## Appendix A: Key File Locations (Inside Container)

| What | Path |
|------|------|
| verl source (editable) | `/root/myCodeLab/host/verl/` |
| mjnemogym source (editable) | `/root/myCodeLab/host/mjnemogym/` |
| vLLM source (editable) | `/opt/vllm/` |
| Training script | `/root/myCodeLab/host/verl/my_scripts/run_moonlight_math_megatron_h200.sh` |
| Runtime env yaml | `/root/myCodeLab/host/verl/my_scripts/my_deepep_env.yaml` |
| Tool schema | `/root/myCodeLab/host/verl/workplace_tools_schema.json` |
| Hydra config | `/root/myCodeLab/host/verl/verl/trainer/config/ppo_megatron_trainer.yaml` |
| Model | `/root/myCodeLab/host/downloads/models/Moonlight16B` |
| Train data | `/root/myCodeLab/host/downloads/datasets/dapo_data/dapo-math-17k.parquet` |
| Test data | `/root/myCodeLab/host/downloads/datasets/dapo_data/aime-2024.parquet` |
| wandb logs | `/root/myCodeLab/host/verl/wandb_my_dirs/` |
| Checkpoints | `/root/myCodeLab/host/verl/ckpts/` |
| dist-packages | `/root/myCodeLab/dist-packages/` → `/usr/local/lib/python3.12/dist-packages` |

## Appendix B: "Do Not Touch" Package Verification

```bash
python3 -c "
pkgs = {
    'torch': '2.10.0a0', 'flash_attn': '2.7.4', 'transformer_engine': '2.12.0',
    'deep_ep': '1.2.1', 'numpy': '1.26.4', 'flashinfer': '0.5.3',
}
import importlib
for pkg, expected_prefix in pkgs.items():
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'unknown')
        status = 'OK' if v.startswith(expected_prefix) else f'MISMATCH (got {v})'
    except ImportError:
        status = 'MISSING'
    print(f'{pkg}: {status}')
"
```
