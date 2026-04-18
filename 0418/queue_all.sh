#!/usr/bin/env bash
# 0418/queue_all.sh — submit all Plan C experiments C1..C9 at once.
#
# Relies on Volcano gang scheduling. Each run is 8 nodes; the cluster has
# ~28 free compute nodes (after nccl-test cleanup), so 3 gangs run in
# parallel and the rest stay Inqueue until capacity frees.
#
# Each submission uses the faster debug-only defaults from submit_debug.sh
# (no pre-train val, no mid-run val, no checkpoint save). The RayJob
# deletes itself on pipeline.sh success (Phase 5), so nodes return to the
# queue as soon as a run terminates.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit_debug.sh"
PIPELINE="$HERE/pipeline.sh"

submit_one() {
    local label="$1"
    local overrides="$2"
    echo "=== $label ==="
    local out name
    # Pass label via DEBUG_LABEL env; submit_debug.sh propagates it through
    # the pod env so the entrypoint writes it to label.txt inside the run dir.
    out=$(DEBUG_LABEL="$label" "$SUBMIT" cdbg5 8 "$overrides" 2>&1) \
        || { echo "FAIL: $label" >&2; echo "$out" >&2; return 1; }
    name=$(echo "$out" | grep -oE 'bra40-dbg-cdbg5-8n-[a-z0-9]+' | head -1)
    if [ -z "$name" ]; then
        echo "FAIL: could not extract RayJob name for $label" >&2
        echo "$out" >&2
        return 1
    fi
    rm -f "/tmp/0418_pipeline_${label}.log"
    nohup bash "$PIPELINE" "$name" "$label" > /dev/null 2>&1 &
    echo "$label -> $name  (monitor PID $!)"
    sleep 2
}

# ── Tier 1 — rollout memory ────────────────────────────────────────────────
submit_one "C1" "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
submit_one "C2" "actor_rollout_ref.rollout.gpu_memory_utilization=0.85 actor_rollout_ref.rollout.free_cache_engine=false"
submit_one "C3" "actor_rollout_ref.rollout.gpu_memory_utilization=0.90 actor_rollout_ref.rollout.free_cache_engine=false"

# ── Tier 2 — drop pipeline parallelism ─────────────────────────────────────
# PP=1 removes pipeline bubbles; 64 GPUs / (1*1*8) = DP=8 replicas of 8 GPU each.
submit_one "C4" "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=1"

# C5 — context parallel fallback if C4 OOMs. Submit anyway; datapoint is useful.
submit_one "C5" "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.context_parallel_size=2 actor_rollout_ref.ref.megatron.context_parallel_size=2"

# ── Tier 3 — offload audit ─────────────────────────────────────────────────
submit_one "C6" "actor_rollout_ref.actor.megatron.optimizer_offload=false actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0.0 actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=false"

# C7 — all offload off. Risky (high OOM). Counter-balance by capping rollout memory.
submit_one "C7" "actor_rollout_ref.actor.megatron.param_offload=false actor_rollout_ref.actor.megatron.optimizer_offload=false actor_rollout_ref.actor.megatron.grad_offload=false actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0.0 actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=false actor_rollout_ref.rollout.gpu_memory_utilization=0.4"

# ── Tier 4 — larger token budget ───────────────────────────────────────────
submit_one "C8" "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48768 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48768 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48768 actor_rollout_ref.rollout.max_num_batched_tokens=48768"

# ── Tier 5 — combined educated guess ───────────────────────────────────────
# Conservative combo: C1 (safe rollout memory) + C4 (no pipeline) + C6 (no opt offload).
# Skips C2/C3 (keepkv, memory-aggressive) and C7 (all-offload-off, OOM risk).
submit_one "C9" "actor_rollout_ref.rollout.gpu_memory_utilization=0.85 actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.optimizer_offload=false actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0.0 actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=false"

echo
echo "All 9 experiments submitted. Watch:"
echo "  HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob -l combo_id=cdbg5"
echo "  HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get podgroup"
