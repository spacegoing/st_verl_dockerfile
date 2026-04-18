#!/usr/bin/env bash
# SUPERSEDED by 0418/queue_all.sh on 2026-04-18. Kept for reference only.
# queue_all.sh submits all 9 C-runs (including C5 + C7 + C9), uses the
# faster debug defaults from submit_debug.sh, and avoids the awk SIGPIPE
# issue this file has.
#
# 0418/queue_tier123.sh — submit all remaining Plan C experiments at once.
#
# Relies on Volcano gang-scheduling to run 2-3 concurrently (8 nodes each)
# on the ~31-node cluster and queue the rest as Inqueue PodGroups.
#
# C1 is assumed already submitted.
#
# Each experiment:
#   - Same combo (cdbg5), same node count (8), same Category-B hparams.
#   - Differs only in Category-A overrides that target compute/memory.
#   - Gets its own pipeline.sh monitor; no cross-chain (we analyze the full
#     set together once all terminate).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit_debug.sh"
PIPELINE="$HERE/pipeline.sh"

# Common settings — empty because submit_debug.sh now applies the debug-only
# defaults (no pre-train val, no mid-run val, no ckpt save) automatically.
# Override here only if an individual experiment needs to RE-enable validation.
COMMON=""

submit_one() {
    local label="$1"
    local overrides="$2"
    echo "=== $label ==="
    local name
    name=$("$SUBMIT" cdbg5 8 "$COMMON $overrides" 2>&1 | awk '/name=/ {sub(/.*name=/,""); print; exit}')
    if [ -z "$name" ]; then
        echo "FAIL: could not submit $label" >&2
        return 1
    fi
    rm -f /tmp/0418_pipeline_${label}.log
    nohup bash "$PIPELINE" "$name" "$label" > /dev/null 2>&1 &
    echo "$label -> $name  (monitor PID $!)"
    sleep 3  # give envsubst+kubectl create time to commit before next submit
}

# Tier 1 — rollout memory
submit_one "C2" "actor_rollout_ref.rollout.gpu_memory_utilization=0.85 actor_rollout_ref.rollout.free_cache_engine=false"
submit_one "C3" "actor_rollout_ref.rollout.gpu_memory_utilization=0.90 actor_rollout_ref.rollout.free_cache_engine=false"

# Tier 2 — parallelism (drop pipeline, expand expert)
# PP=1 removes pipeline bubbles. Total GPU per DP replica = PP*TP*EP = 1*1*8 = 8.
# With 64 GPUs total → DP = 8 (each DP replica on one node). Keep EP=8.
submit_one "C4" "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=1"

# Tier 3 — offload audit: keep optimizer on GPU
submit_one "C6" "actor_rollout_ref.actor.megatron.optimizer_offload=false actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0.0 actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=false"

# Tier 4 — bigger token budget per microbatch (pack 2 full seqs)
submit_one "C8" "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48768 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48768 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48768 actor_rollout_ref.rollout.max_num_batched_tokens=48768"

echo
echo "All Tier 1-4 experiments submitted. Watch with:"
echo "  HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob -l combo_id=cdbg5,training-type=40bra-dbg"
echo "  HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get podgroup"
