# DeepEP / NVSHMEM IBGDA Failure Report — B300 Cluster (b32/b31)

**Date:** 2026-03-23
**Cluster:** b32 (10.12.11.6) + b31 (10.12.11.5), 2×8 NVIDIA B300 SXM6 AC
**NICs:** mlx5_10–mlx5_17 (RoCEv2, GID index 3, TC=138)
**NVSHMEM:** v3.4.5
**Software:** NVIDIA NeMo 26.02 container, DeepEP (deep_ep), Megatron-Bridge

We are running 2-node GRPO training using the [DeepEP](https://github.com/deepseek-ai/DeepEP) MoE expert-parallel dispatcher, which uses NVSHMEM IBGDA for cross-node all-to-all communication. We hit two layered failures that forced us to fall back to the slower NCCL alltoall dispatcher. Both failures point to IB fabric configuration issues on the cluster side.

---

## Issue 1: DCT QPs Not Supported (Immediate Failure)

### Symptom

NVSHMEM IBGDA fails immediately during initialization on both nodes when using the default `gpu` NIC handler (which requires DCT — Dynamically Connected Transport — QPs):

```
ibgda.cpp:2234: NULL value Unable to create ah.
ibgda.cpp:2966: non-zero status: 7 create DCT share err.
transport.cpp:420: non-zero status: 7 connect EPS failed
init.cu:1047: non-zero status: 7 nvshmem setup connections failed
```

Status 7 = `EPERM` / resource error at the IB verbs layer — `ibv_create_ah()` returns NULL, DCT QP creation fails on every rank on both nodes.

### Minimal Reproduction

Run on b32 (requires SSH access to b31, both must have the `vrl` container running):

```bash
# File: verl/my_scripts/run_deepep_test.sh
# Run with: bash run_deepep_test.sh gpu
bash /mnt/public/lichang93/st_verl_dockerfile/verl/my_scripts/run_deepep_test.sh gpu
```

This launches 16 torchrun processes (8 per node) with:

```bash
NVSHMEM_BOOTSTRAP=IB \
NVSHMEM_IB_ENABLE_IBGDA=1 \
NVSHMEM_IB_GID_INDEX=3 \
NVSHMEM_HCA_LIST=mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17 \
NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1 \
NVSHMEM_IBGDA_NIC_HANDLER=gpu \
NVSHMEM_IB_TRAFFIC_CLASS=96 \
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=10.12.11.6 --master_port=29500 \
  verl/my_scripts/test_deepep_internode.py
```

**Actual error log:** `verl/logs/deepep_test_b32_20260323_111016.log` (and matching b31 log)

### Root Cause

`NVSHMEM_IBGDA_NIC_HANDLER=gpu` uses IBGDA with DCT (Dynamically Connected Transport) QPs — a specific InfiniBand QP type that allows one-sided RDMA without pre-established connections. The IB fabric on this cluster **does not support DCT QP creation** (`ibv_exp_create_qp` or `mlx5dv_create_qp` with `IBV_QPT_DRIVER` / DCT type returns error).

### Workaround Applied

Switching to `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory` uses IBRC (Reliable Connected) QPs instead of DCT. The standalone test passes with this setting:

```bash
bash run_deepep_test.sh cpu   # PASSED — log: deepep_test_b32_20260323_133434.log
```

---

## Issue 2: IB QP Resource Exhaustion When NCCL and NVSHMEM Coexist

### Symptom

Even with `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory`, NVSHMEM IBGDA initialization **fails at training step 1** inside Megatron training — but **passes in the standalone test** (same hardware, same settings).

The error is the same:
```
ibgda.cpp:2966: non-zero status: 7 create DCT share err.
transport.cpp:420: non-zero status: 7 connect EPS failed
init.cu:1047: non-zero status: 7 nvshmem setup connections failed
```

### What Is Different Between the Passing and Failing Cases

| Scenario | NCCL initialized? | NVSHMEM IBGDA init | Result |
|---|---|---|---|
| Standalone DeepEP test (`run_deepep_test.sh cpu`) | **No** | At startup | **PASS** |
| Megatron training with DeepEP flex dispatcher | **Yes** (before NVSHMEM) | Lazily at step 1 | **FAIL** |

### Minimal Reproduction Sequence

1. Start 16 torchrun processes across 2 nodes and **let NCCL initialize** its process group (standard `torch.distributed.init_process_group`). NCCL will create RC QPs on all IB NICs for all rank-pairs.
2. **Then** attempt to initialize NVSHMEM IBGDA (`nvshmemx_init_attr` with IBGDA transport).
3. NVSHMEM will fail with status 7 — same DCT/QP error as above.

In Megatron, NCCL initializes during `WorkerDict` actor startup (step 0). NVSHMEM is initialized lazily inside `deep_ep.Buffer()` at the first MoE dispatch call (step 1, `compute_log_prob`). By then, NCCL has already allocated IB resources.

### Resource Math

NCCL QP allocation for a 16-rank EP group across 2 nodes:
- 16 EP processes × 8 NICs (`mlx5_10`–`mlx5_17`) × `NCCL_IB_QPS_PER_CONNECTION=8` = **1024 RC QPs** just for NCCL

When NVSHMEM IBGDA then tries to create its own QPs (even IBRC, not DCT), the fabric QP table or per-process context limit is exhausted → `ibv_create_ah` / `ibv_create_qp` returns NULL → status 7.

### Root Cause (Our Assessment)

This is an **IB fabric resource limit** — either:

1. **Per-port or per-HCA QP limit** is too low for the combined NCCL + NVSHMEM workload on these NICs.
2. **DCT QP table exhaustion**: even though we're using `cpu_host_memory` (IBRC), NVSHMEM IBGDA internally still attempts DCT-related resource allocation during transport initialization on mlx5 NICs.
3. **`ibv_create_ah` failure**: the address handle creation (`Unable to create ah`) that precedes the DCT error suggests the IB UD (Unreliable Datagram) QP used for bootstrapping is also failing — possibly a multicast group or LID resolution issue on the fabric.

---

## Evidence Summary

| Test | Setting | NCCL running | Result |
|---|---|---|---|
| `run_deepep_test.sh gpu` | `NIC_HANDLER=gpu` (DCT) | No | FAIL — DCT not supported |
| `run_deepep_test.sh cpu` | `NIC_HANDLER=cpu_host_memory` | No | PASS |
| Megatron training (flex) | `NIC_HANDLER=cpu_host_memory` | **Yes** | FAIL at step 1 — IB QP exhaustion |
| Megatron training (alltoall) | NVSHMEM not used | Yes | PASS — fallback, no IBGDA |

Logs for the failing and passing runs:
- Failing (DCT): `verl/logs/deepep_test_b32_20260323_111016.log`, `deepep_test_b31_20260323_111016.log`
- Passing (cpu, standalone): `verl/logs/deepep_test_b32_20260323_133434.log`, `deepep_test_b31_20260323_133434.log`

---

## CLIs to Diagnose

Run these on b32 and b31 (inside or outside the container — `ibv_devinfo`, `ibstat` need the host IB stack):

### 1. Check DCT support on each NIC

```bash
# DCT is an mlx5 experimental verb. Check if the NIC advertises it:
ibv_devinfo -d mlx5_10 -v 2>/dev/null | grep -i "dct\|dc_ini\|dc_type\|transport"

# Alternatively, check via mlx5 device capabilities:
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo -n "$dev: "; ibv_devinfo -d $dev 2>/dev/null | grep -i "fw_ver\|node_guid" | head -1
done
```

### 2. Check current QP usage and limits

```bash
# Total QP limit per HCA (kernel reports)
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo -n "$dev max_qp: "
    cat /sys/class/infiniband/${dev}/ports/1/counters/../../.. 2>/dev/null || true
    ibv_devinfo -d $dev 2>/dev/null | grep -E "max_qp[^_]|max_cq[^_]" | head -3
done

# Current QP usage via rdma tools
rdma res show qp 2>/dev/null | wc -l
rdma res show qp 2>/dev/null | head -20
```

### 3. Check GID table and RoCEv2 config

```bash
# GID index 3 must be RoCEv2 (type 'RoCE v2') on both nodes
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo "=== $dev ==="
    cat /sys/class/infiniband/${dev}/ports/1/gid_attrs/types/3 2>/dev/null
    cat /sys/class/infiniband/${dev}/ports/1/gids/3 2>/dev/null
done
```

### 4. Check IB port state and link layer

```bash
ibstat 2>/dev/null | grep -A5 "CA '\\|Port 1"
# All ports should show: State: Active, Physical state: LinkUp, Link layer: Ethernet (RoCE)
```

### 5. Check address handle / UD QP creation capability

```bash
# The 'Unable to create ah' error suggests UD AH creation fails.
# This can be caused by missing multicast group membership or routing issues.
# Check if AH creation works for a known GID:
python3 -c "
import pyverbs.device as d, pyverbs.pd as pd, pyverbs.cq as cq, pyverbs.qp as qp, pyverbs.addr as addr
ctx = d.Context(name='mlx5_10')
p = pd.PD(ctx)
# Try creating a UD address handle to GID index 3
ah_attr = addr.AHAttr(gr=addr.GlobalRoute(dgid=addr.GID('::1'), sgid_index=3), is_global=1, port_num=1)
ah = addr.AH(p, attr=ah_attr)
print('AH creation: OK')
" 2>&1
```

### 6. Check per-process file descriptor and resource limits

```bash
# NVSHMEM IBGDA opens many FDs (one per QP per NIC). Check limits:
ulimit -n          # open files limit (should be >=65536)
cat /proc/sys/fs/file-max   # system-wide limit

# Check current FD usage in the training container:
ls /proc/$(pgrep -f torchrun | head -1)/fd 2>/dev/null | wc -l
```

### 7. Check IB fabric for DCT QP type support

```bash
# mlx5 DCT support can be queried via the vendor query device extended:
python3 -c "
import ctypes
lib = ctypes.CDLL('libibverbs.so.1')
# Or use pyverbs if available:
try:
    from pyverbs.device import Context
    from pyverbs.enums import ibv_device_attr_ex
    ctx = Context(name='mlx5_10')
    attrs = ctx.query_device_ex()
    print(attrs)
except Exception as e:
    print(f'Error: {e}')
" 2>&1

# Simpler: check if mlx5dv_create_qp with IBV_QPT_DRIVER works:
# (This is what NVSHMEM IBGDA tries internally for DCT)
ibv_rc_pingpong -d mlx5_10 -g 3 &   # quick sanity check for basic RC QP
```

### 8. Check if NCCL QP count is the culprit (before vs after NCCL init)

```bash
# Run this before and after starting the training container to see QP delta:
rdma res show qp 2>/dev/null | wc -l
# Expected delta after NCCL init (16 ranks × 8 NICs × 8 QPS_PER_CONNECTION):
#   ~1024 QPs across all 8 NICs = ~128 per NIC
```

---

## What We Need From the Admin

1. **Confirm whether DCT QPs are supported** on the mlx5 NICs in this cluster. If not, is this a firmware/driver config issue that can be fixed, or a hardware limitation?

2. **Check the per-HCA QP limit** and whether it can be raised. With `NCCL_IB_QPS_PER_CONNECTION=8` across 16 EP ranks and 8 NICs, NCCL alone uses ~128 QPs per NIC. If NVSHMEM needs additional QPs, the limit may need to be raised via:
   ```bash
   # mlx5 QP limit is often set in firmware or via mlnx_qos / mlxconfig:
   mlxconfig -d /dev/mst/mt... query | grep -i qp
   ```

3. **IB fabric routing**: confirm that UD address handles (for NVSHMEM bootstrap over IB) can be created between b32 and b31. The `Unable to create ah` error before the DCT error suggests the IB UD path might also be broken.

4. If DCT is unsupported by design, we need guidance on whether `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory` with IBRC QPs is expected to work at scale (16 ranks + existing NCCL QPs) on this fabric, or if there are QP reservation/prioritization options.

---

## Our Current Workaround

We switched from DeepEP flex dispatcher to NCCL alltoall (`moe_token_dispatcher_type=alltoall`, `moe_enable_deepep=False`). This avoids NVSHMEM entirely and training works correctly (`timing_s/step: 342s` for Moonlight-16B, 2×8 B300). The performance cost vs DeepEP flex is acceptable for smoke tests but we would like DeepEP to work for production runs.
