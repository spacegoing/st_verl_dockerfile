#!/usr/bin/env python3
"""Minimal Ray GPU-burn app for kuberay-volcano gang-scheduling verification.

Each RayJob has 8 pods (1 head + 7 workers), each pod owning 8 GPUs on its own
B300 node. We dispatch 8 parallel @ray.remote(num_gpus=8) tasks. Ray can only
fit one such task per node (each needs 8 GPUs, each node has 8), so the tasks
distribute 1-to-1 across the 8 nodes. Each task does torch.matmul in a loop
for DURATION_S seconds (default 300 = 5 min) and logs to PFS.

Invoked via RayJob entrypoint:
    python3 /root/myCodeLab/host/voltest/gpu_burn.py
"""
import os
import socket
import sys
import time

import ray
import torch


@ray.remote(num_gpus=8)
def gpu_burn(job_name: str, node_idx: int, duration_s: int, log_path: str):
    """Run GEMM on all 8 local GPUs for `duration_s` seconds; append to `log_path`."""
    host = socket.gethostname()
    n_gpu = torch.cuda.device_count()

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] [{job_name}] [node{node_idx}] [{host}] {msg}"
        print(line, flush=True)
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"WARN: log write failed: {e}", flush=True)

    log(f"gpu_burn start: duration={duration_s}s, GPUs visible={n_gpu}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}")
    log(f"torch.version.cuda={torch.version.cuda}, torch.__version__={torch.__version__}")
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        log(f"  GPU[{i}] = {name}")

    # One 4096x4096 tensor per GPU
    tensors = [torch.randn(4096, 4096, device=f"cuda:{i}") for i in range(n_gpu)]
    for t in tensors:
        torch.cuda.synchronize(t.device)

    start = time.time()
    iter_count = 0
    last_log = start
    while time.time() - start < duration_s:
        for i in range(n_gpu):
            tensors[i] = tensors[i] @ tensors[i]
            tensors[i] = tensors[i] / (tensors[i].abs().max() + 1e-6)
        iter_count += 1
        now = time.time()
        if now - last_log >= 30.0:
            elapsed = now - start
            log(f"iter={iter_count} elapsed={elapsed:.1f}s rate={iter_count/elapsed:.2f} it/s")
            last_log = now

    for t in tensors:
        torch.cuda.synchronize(t.device)
    elapsed = time.time() - start
    log(f"gpu_burn DONE: total_iter={iter_count} total_elapsed={elapsed:.1f}s")
    return {"host": host, "node_idx": node_idx, "iters": iter_count, "elapsed_s": elapsed}


def main():
    job_name = os.environ.get("JOB_NAME", "voltest-unknown")
    duration_s = int(os.environ.get("DURATION_S", "300"))
    num_nodes = int(os.environ.get("NUM_NODES", "8"))
    log_dir = os.environ.get("LOG_DIR", "/root/myCodeLab/host/voltest/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{job_name}.log")

    print(f"[main] job_name={job_name} duration_s={duration_s} num_nodes={num_nodes} log_path={log_path}", flush=True)

    ray.init(address="auto")
    resources = ray.cluster_resources()
    total_gpus = resources.get("GPU", 0)
    print(f"[main] Ray cluster resources: {resources}", flush=True)

    expected_gpus = num_nodes * 8
    if total_gpus < expected_gpus:
        print(f"[main] ERROR: cluster has {total_gpus} GPUs, need {expected_gpus}; aborting", flush=True)
        sys.exit(1)

    # Dispatch num_nodes parallel tasks; Ray spreads them (one per node, since
    # each task needs 8 GPUs and each node provides exactly 8)
    print(f"[main] Dispatching {num_nodes} gpu_burn tasks", flush=True)
    futures = [gpu_burn.remote(job_name, i, duration_s, log_path) for i in range(num_nodes)]
    results = ray.get(futures)
    print(f"[main] All {num_nodes} tasks finished:", flush=True)
    for r in results:
        print(f"  {r}", flush=True)


if __name__ == "__main__":
    main()
