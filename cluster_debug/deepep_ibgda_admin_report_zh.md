# DeepEP / NVSHMEM IBGDA 故障报告 — B300 集群 (b32/b31)

**日期：** 2026-03-23
**集群：** b32 (10.12.11.6) + b31 (10.12.11.5)，2×8 NVIDIA B300 SXM6 AC
**网卡：** mlx5_10–mlx5_17（RoCEv2，GID index 3，TC=138）
**NVSHMEM 版本：** v3.4.5
**软件栈：** NVIDIA NeMo 26.02 容器，DeepEP（deep_ep），Megatron-Bridge

我们在进行 2 节点 GRPO 训练，使用 [DeepEP](https://github.com/deepseek-ai/DeepEP) MoE 专家并行调度器，依赖 NVSHMEM IBGDA 进行跨节点 all-to-all 通信。遇到两个叠加的故障，最终不得不回退到较慢的 NCCL alltoall 调度器。两个故障均指向集群侧的 IB 网络配置问题。

---

## 问题一：不支持 DCT QP（立即报错）

### 现象

使用默认的 `gpu` NIC handler（依赖 DCT——Dynamically Connected Transport——QP）时，NVSHMEM IBGDA 在初始化阶段立即失败，两个节点均报错：

```
ibgda.cpp:2234: NULL value Unable to create ah.
ibgda.cpp:2966: non-zero status: 7 create DCT share err.
transport.cpp:420: non-zero status: 7 connect EPS failed
init.cu:1047: non-zero status: 7 nvshmem setup connections failed
```

状态码 7 = IB verbs 层的 `EPERM`/资源错误，`ibv_create_ah()` 返回 NULL，DCT QP 创建失败，两节点所有 rank 均如此。

### 最小复现脚本

在 b32 上执行（需要能 SSH 到 b31，两台机器均需运行 `vrl` 容器）：

```bash
# 脚本位置：verl/my_scripts/run_deepep_test.sh
bash /mnt/public/lichang93/st_verl_dockerfile/verl/my_scripts/run_deepep_test.sh gpu
```

该脚本在两个节点各启动 8 个 torchrun 进程（共 16 个），关键环境变量如下：

```bash
NVSHMEM_BOOTSTRAP=IB \
NVSHMEM_IB_ENABLE_IBGDA=1 \
NVSHMEM_IB_GID_INDEX=3 \
NVSHMEM_HCA_LIST=mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17 \
NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1 \
NVSHMEM_IBGDA_NIC_HANDLER=gpu \       # <-- 触发 DCT
NVSHMEM_IB_TRAFFIC_CLASS=96 \
torchrun --nproc_per_node=8 --nnodes=2 ...
```

**实际报错日志：**
- `verl/logs/deepep_test_b32_20260323_111016.log`
- `verl/logs/deepep_test_b31_20260323_111016.log`

### 根因

`NVSHMEM_IBGDA_NIC_HANDLER=gpu` 使用 IBGDA + DCT QP——一种允许单边 RDMA 无需预建连接的 InfiniBand QP 类型。**本集群 IB 网络不支持 DCT QP 创建**（`ibv_exp_create_qp` 或 `mlx5dv_create_qp` 以 DCT 类型调用时返回错误）。

### 已应用的规避措施

切换为 `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory`，改用 IBRC（Reliable Connected）QP，standalone 测试通过：

```bash
bash run_deepep_test.sh cpu   # 通过 — 日志：deepep_test_b32_20260323_133434.log
```

---

## 问题二：NCCL 与 NVSHMEM 共存时 IB QP 资源耗尽

### 现象

即使设置了 `NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory`，在 Megatron 训练的**第 1 步**，NVSHMEM IBGDA 初始化仍然失败——但**standalone 测试**（相同硬件、相同配置）**通过**。

报错内容与问题一相同：
```
ibgda.cpp:2966: non-zero status: 7 create DCT share err.
transport.cpp:420: non-zero status: 7 connect EPS failed
init.cu:1047: non-zero status: 7 nvshmem setup connections failed
```

### 通过与失败场景的差异

| 场景 | NCCL 是否已初始化 | NVSHMEM IBGDA 初始化时机 | 结果 |
|---|---|---|---|
| standalone DeepEP 测试（`run_deepep_test.sh cpu`） | **否** | 启动时 | **通过** |
| Megatron 训练（DeepEP flex 调度器） | **是**（NVSHMEM 之前） | 第 1 步延迟初始化 | **失败** |

### 复现步骤（最小）

1. 跨两节点启动 16 个 torchrun 进程，**让 NCCL 初始化**进程组（`torch.distributed.init_process_group`）。NCCL 会在所有 IB 网卡上为所有 rank 对创建 RC QP。
2. **之后**再尝试初始化 NVSHMEM IBGDA（`nvshmemx_init_attr` 使用 IBGDA transport）。
3. NVSHMEM 将以 status 7 失败，报错与上面相同。

在 Megatron 中：NCCL 在 `WorkerDict` actor 启动时初始化（第 0 步）；NVSHMEM 在第 1 步第一次 MoE dispatch（`compute_log_prob`）时延迟创建 `deep_ep.Buffer()`。此时 NCCL 已占用 IB 资源。

### 资源估算

16 rank EP 组跨 2 节点的 NCCL QP 分配量：
- 16 EP 进程 × 8 网卡（`mlx5_10`–`mlx5_17`）× `NCCL_IB_QPS_PER_CONNECTION=8` = **1024 个 RC QP**（仅 NCCL 部分）

NVSHMEM IBGDA 再尝试创建自己的 QP 时（即便是 IBRC，非 DCT），IB QP 表或进程级上下文已耗尽 → `ibv_create_ah` / `ibv_create_qp` 返回 NULL → status 7。

### 根因分析（我们的判断）

这是一个 **IB 网络资源限制**问题，可能原因：

1. **单 HCA QP 数量上限过低**，无法同时承载 NCCL + NVSHMEM 的 QP 需求。
2. **DCT QP 表耗尽**：即使使用 `cpu_host_memory`（IBRC），NVSHMEM IBGDA 在 mlx5 网卡上初始化 transport 时，内部仍可能尝试 DCT 相关资源分配。
3. **`ibv_create_ah` 失败**：DCT 报错之前出现 `Unable to create ah`（地址句柄创建失败），表明用于 bootstrap 的 IB UD QP 也可能存在问题——可能是 fabric 上的组播组或 LID 解析问题。

---

## 证据汇总

| 测试 | 配置 | NCCL 是否运行 | 结果 |
|---|---|---|---|
| `run_deepep_test.sh gpu` | `NIC_HANDLER=gpu`（DCT） | 否 | 失败——不支持 DCT |
| `run_deepep_test.sh cpu` | `NIC_HANDLER=cpu_host_memory` | 否 | 通过 |
| Megatron 训练（flex 调度器） | `NIC_HANDLER=cpu_host_memory` | **是** | 第 1 步失败——IB QP 耗尽 |
| Megatron 训练（alltoall 调度器） | 不使用 NVSHMEM | 是 | 通过（回退方案） |

相关日志文件：
- 失败（DCT）：`verl/logs/deepep_test_b32_20260323_111016.log`，`deepep_test_b31_20260323_111016.log`
- 通过（cpu，standalone）：`verl/logs/deepep_test_b32_20260323_133434.log`，`deepep_test_b31_20260323_133434.log`

---

## 诊断命令

以下命令请在 b32 和 b31 上分别执行（`ibv_devinfo`、`ibstat` 等需在宿主机或容器内有 IB 库访问权限）：

### 1. 检查各网卡是否支持 DCT QP

```bash
# DCT 是 mlx5 实验性 verb，检查网卡是否声明支持：
ibv_devinfo -d mlx5_10 -v 2>/dev/null | grep -i "dct\|dc_ini\|dc_type\|transport"

# 批量查看固件版本（确认两节点驱动/固件一致）：
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo -n "$dev: "; ibv_devinfo -d $dev 2>/dev/null | grep -i "fw_ver\|node_guid" | head -1
done
```

### 2. 检查当前 QP 数量及上限

```bash
# 查看每块网卡的 max_qp 限制：
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo "=== $dev ==="
    ibv_devinfo -d $dev 2>/dev/null | grep -E "max_qp[^_]|max_cq[^_]|max_mr" | head -5
done

# 通过 rdma 工具查看当前 QP 数（训练前 vs 训练中对比）：
rdma res show qp 2>/dev/null | wc -l
rdma res show qp 2>/dev/null | head -30
```

### 3. 检查 GID 表和 RoCEv2 配置

```bash
# GID index 3 在两节点上均须为 RoCEv2（类型显示 "RoCE v2"）：
for dev in mlx5_10 mlx5_11 mlx5_12 mlx5_13 mlx5_14 mlx5_15 mlx5_16 mlx5_17; do
    echo "=== $dev GID[3] ==="
    cat /sys/class/infiniband/${dev}/ports/1/gid_attrs/types/3 2>/dev/null
    cat /sys/class/infiniband/${dev}/ports/1/gids/3 2>/dev/null
done
```

### 4. 检查 IB 端口状态

```bash
ibstat 2>/dev/null | grep -A8 "CA '\\|Port 1"
# 所有端口应显示：State: Active，Physical state: LinkUp，Link layer: Ethernet（RoCE）
```

### 5. 检查地址句柄（AH）创建是否可用

```bash
# 'Unable to create ah' 说明 UD AH 创建失败，可能是组播/路由问题。
# 快速验证 RC QP 基础功能：
ibv_rc_pingpong -d mlx5_10 -g 3 &   # 服务端
ibv_rc_pingpong -d mlx5_10 -g 3 <b31_gid>  # 客户端（在 b31 执行）
```

### 6. 检查 QP 数量上限是否可调（mlxconfig）

```bash
# 查看固件层 QP 配置（需要 mst 工具）：
mst start 2>/dev/null
for dev in $(mst status 2>/dev/null | grep mt | awk '{print $1}'); do
    echo "=== $dev ==="
    mlxconfig -d $dev query 2>/dev/null | grep -i "log_max_qp\|num_qp\|qp"
done
```

### 7. 检查文件描述符限制

```bash
# NVSHMEM IBGDA 每个 QP/NIC 都需要 FD。检查限制：
ulimit -n                         # 当前 shell open files 限制（建议 >=65536）
cat /proc/sys/fs/file-max         # 系统级限制

# 容器内进程的实际 FD 使用量：
docker exec vrl bash -c 'ls /proc/$(pgrep -f torchrun | head -1)/fd 2>/dev/null | wc -l'
```

---

## 我们希望管理员协助确认的内容

1. **本集群 mlx5 网卡是否支持 DCT QP？** 如不支持，是固件/驱动配置问题还是硬件本身不支持？是否可以通过 `mlxconfig` 或升级固件来开启？

2. **单 HCA 的 QP 数量上限是否可以提高？** NCCL（1024 QP）+ NVSHMEM IBGDA 的 QP 加在一起可能超出每块网卡的限制。可否通过以下方式调整：
   ```bash
   mlxconfig -d /dev/mst/<device> set LOG_MAX_QP=<value>
   ```

3. **UD 地址句柄（AH）创建失败的原因**。`ibv_create_ah` 在 GID index 3 上失败，需确认两节点间 IB UD 路径是否正常（组播组成员、SM（子网管理器）配置等）。

4. 如果 DCT 设计上不受支持，能否说明在 `cpu_host_memory` 模式下，16 rank + 已有 NCCL QP 的情况下 NVSHMEM IBGDA 是否预期可以工作？是否有 QP 预留或优先级配置可供参考？

---

## 当前已用规避方案

我们将 DeepEP flex 调度器切换为 NCCL alltoall（`moe_token_dispatcher_type=alltoall`，`moe_enable_deepep=False`），完全绕过 NVSHMEM。训练已正常运行（Moonlight-16B，2×8 B300，每步约 342 秒）。Smoke test 下性能损失可接受，但我们希望在生产训练中使用 DeepEP 以获得更好的 MoE dispatch 性能，因此需要解决上述问题。
