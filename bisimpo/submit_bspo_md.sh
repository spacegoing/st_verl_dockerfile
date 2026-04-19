#!/usr/bin/env bash
# bisimpo/submit_bspo_md.sh — launch a multi-domain BSPO RayJob.
#
# Forked from submit_bspo.sh (single-domain) with:
#   - PROD_SUBMIT points to submit_md.sh (forks rayjob_md.yaml)
#   - per-combo hparams for cdbgmd* (debug) and cbsp501..cbsp505 (formal)
#   - NNODES default: 16 for formal, 4 for debug
#
# Usage:
#   ./submit_bspo_md.sh cdbgmd1                 # debug smoke (4 nodes, 1 step)
#   ./submit_bspo_md.sh cbsp501                 # V1 multi-domain formal (16 nodes, 120 steps)
#   NNODES=4 ./submit_bspo_md.sh cbsp501        # run V1 formal on 4 nodes instead
#
# Set BSPO_DEBUG=1 to force the debug override pack at the entrypoint level.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_SUBMIT="$HERE/../iter_kuberay_32nodes_verl_training/submit/submit_md.sh"

COMBO_ID="${1:?usage: $0 <combo_id> [extra Hydra overrides]}"
shift
EXTRA_USER="${*:-}"

# ── Per-combo BSPO hparams ────────────────────────────────────────────────────
case "$COMBO_ID" in
    # ── debug smoke combos (fast-fail) ──
    cbdg-md-v1-smoke)  VAR=simplest;           DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cbdg-md-v2-smoke)  VAR=hierarchy;          DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cbdg-md-v3-smoke)  VAR=strict_wasserstein; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cbdg-md-v4-smoke)  VAR=w_penalty;          DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cbdg-md-v5-smoke)  VAR=w_penalty_only;     DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;

    # ── formal V1..V5 multi-domain runs ──
    cbmd-v1)           VAR=simplest;         DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbmd-v2)           VAR=hierarchy;        DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbmd-v3)           VAR=strict_wasserstein; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbmd-v4)           VAR=w_penalty;        DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbmd-v5)           VAR=w_penalty_only;   DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;

    *)  echo "error: unknown BSPO-MD combo '$COMBO_ID'" >&2; exit 2 ;;
esac

# ── BSPO override string ──────────────────────────────────────────────────────
# Multi-domain uses loss_mode=bspo_md (separate fn from single-domain 'bspo').
BSPO_OVR=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=bspo_md"
    "actor_rollout_ref.actor.bspo_variant=$VAR"
    "actor_rollout_ref.actor.bspo_delta=$DELTA"
    "actor_rollout_ref.actor.bspo_lambda_tj=$LAMBDA"
)

ABLATE_OVR=(
    "trainer.test_freq=10"
    "trainer.val_before_train=true"
    "trainer.save_freq=9999"
    "trainer.log_val_generations=0"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP"
)

ALL_OVR="${BSPO_OVR[*]} ${ABLATE_OVR[*]} ${EXTRA_USER}"

# Debug combos default to 2 nodes + BSPO_DEBUG=1 (minimum viable shape);
# formal default to 4 nodes (C11 profile, 7-concurrent budget on 28 nodes).
case "$COMBO_ID" in
    cbdg-*)    DEFAULT_NNODES=2; DEFAULT_DEBUG=1 ;;
    *)         DEFAULT_NNODES=4; DEFAULT_DEBUG=0 ;;
esac

echo "[submit_bspo_md] combo=$COMBO_ID variant=$VAR delta=$DELTA lambda=$LAMBDA"
echo "[submit_bspo_md] overrides: $ALL_OVR"

NNODES="${NNODES:-$DEFAULT_NNODES}" \
BSPO_DEBUG="${BSPO_DEBUG:-$DEFAULT_DEBUG}" \
    exec "$PROD_SUBMIT" "$COMBO_ID" "$ALL_OVR"
