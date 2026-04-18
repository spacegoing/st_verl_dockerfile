#!/usr/bin/env bash
# 0418/submit_debug.sh — launch a parameterized debug RayJob for sharding experiments.
#
# Usage:
#   ./submit_debug.sh <combo_id> <nnodes> [<extra Hydra overrides> ...]
#   ./submit_debug.sh cdbg5 16
#   ./submit_debug.sh cdbg5 8  'data.val_max_samples=32 trainer.test_freq=3'
#   ./submit_debug.sh cdbg5 4
#
# What it does:
#   - Validates combo id format ^c[a-z]*[0-9]+$  (accepts cdbg5, c351, etc.)
#   - Computes WORKER_REPLICAS = NNODES - 1
#   - envsubsts placeholders in 0418/rayjob_debug.yaml
#   - kubectl create (generateName picks a unique suffix)
#   - Prints follow-up watch/log commands

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL="${HERE}/rayjob_debug.yaml"

unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export NO_PROXY="${K8S_API_IP:-180.184.249.201}" \
       no_proxy="${K8S_API_IP:-180.184.249.201}"

COMBO_ID="${1:?usage: $0 <combo_id> <nnodes> [extra Hydra overrides ...]}"
NNODES="${2:?usage: $0 <combo_id> <nnodes> [extra Hydra overrides ...]}"
shift 2 || true

# Optional human label (e.g. "C1") — propagated through the pod env so the
# entrypoint writes it to $RUN_DIR/label.txt for later correlation. If unset,
# the run dir simply has no label.txt.
DEBUG_LABEL="${DEBUG_LABEL:-}"

# ─── DEBUG-ONLY defaults ─────────────────────────────────────────────────────
# These are safe for debug runs because none of them affect training numerics
# (all Category A per 0418/plan_c_compute_opt.md Section 1). They strip the
# validation and checkpoint overhead so a 5-step run completes faster without
# changing the training-trajectory math.
#
# lr_warmup_steps=2     : env_k8s_b300_16node.yaml hard-codes 10, which
#                         trips `assert warmup < decay` when total_training_steps=5.
# val_before_train=false: skip the pre-train baseline validation (~3-10 min).
# test_freq=9999        : skip all mid-run validation; training trajectory
#                         equivalence is verified from per-step critic_score
#                         in perf_log.jsonl, not from val accuracy.
# save_freq=9999        : no checkpoint writes during the debug run.
# log_val_generations=0 : skip the text-generation samples in wandb/log.
# val_max_samples=32    : if anyone re-enables val via an explicit override,
#                         keep it small so it stays cheap.
#
# Any of these can be overridden by passing them in EXTRA_OVERRIDES (Hydra
# uses last-wins, so user overrides take precedence).
DEBUG_DEFAULTS=(
    "actor_rollout_ref.actor.optim.lr_warmup_steps=2"
    "trainer.val_before_train=false"
    "trainer.test_freq=9999"
    "trainer.save_freq=9999"
    "trainer.log_val_generations=0"
    "data.val_max_samples=32"
)

EXTRA_OVERRIDES="${DEBUG_DEFAULTS[*]} ${*:-}"

[[ "$COMBO_ID" =~ ^c[a-z]*[0-9]+$ ]] || {
    echo "error: invalid combo_id '$COMBO_ID' (expected c<id>; e.g. cdbg5 or c351)" >&2
    exit 2
}
[[ "$NNODES" =~ ^[0-9]+$ ]] || {
    echo "error: invalid nnodes '$NNODES' (expected positive integer)" >&2
    exit 2
}
(( NNODES >= 1 && NNODES <= 32 )) || {
    echo "error: nnodes must be in [1,32]" >&2
    exit 2
}
[ -r "$TPL" ] || { echo "error: template missing: $TPL" >&2; exit 3; }

WORKER_REPLICAS=$(( NNODES - 1 ))

export COMBO_ID NNODES WORKER_REPLICAS EXTRA_OVERRIDES DEBUG_LABEL

RAYJOB_NAME=$(envsubst '${COMBO_ID} ${NNODES} ${WORKER_REPLICAS} ${EXTRA_OVERRIDES} ${DEBUG_LABEL}' < "$TPL" \
              | kubectl create -f - -o name \
              | sed 's|^rayjob.ray.io/||')

echo "[submit_debug] combo=$COMBO_ID  nnodes=$NNODES  worker_replicas=$WORKER_REPLICAS"
echo "[submit_debug] name=$RAYJOB_NAME"
[ -n "$EXTRA_OVERRIDES" ] && echo "[submit_debug] overrides: $EXTRA_OVERRIDES"
echo
echo "watch progress:"
echo "  kubectl get rayjob $RAYJOB_NAME -w"
echo "podgroup (should appear within 5s, minMember=$NNODES):"
echo "  kubectl get podgroup -l ray.io/rayjob=$RAYJOB_NAME"
echo "head pod logs (once head is Running):"
echo "  kubectl logs -l ray.io/cluster=\$(kubectl get rayjob $RAYJOB_NAME -o jsonpath='{.status.rayClusterName}') -c ray-head -f"
echo "verify gang scheduling now:"
echo "  $HERE/verify_gang.sh $RAYJOB_NAME"
