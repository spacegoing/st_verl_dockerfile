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
    # ── debug combos (fast-fail smoke) ──
    cdbgmd1)      VAR=simplest;         DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cdbgmd4)      VAR=w_penalty;        DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cdbgmd5)      VAR=w_penalty_only;   DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;

    # ── formal V1..V5 multi-domain runs ──
    cbsp501)      VAR=simplest;         DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbsp502)      VAR=hierarchy;        DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # needs S3 code
    cbsp503)      VAR=strict_wasserstein; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # needs S3 code
    cbsp504)      VAR=w_penalty;        DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbsp505)      VAR=w_penalty_only;   DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;

    *)  echo "error: unknown BSPO-MD combo '$COMBO_ID'" >&2; exit 2 ;;
esac

# ── BSPO override string ──────────────────────────────────────────────────────
BSPO_OVR=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=bspo"
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

# Debug combos default to 4 nodes + BSPO_DEBUG=1; formal default to 16.
case "$COMBO_ID" in
    cdbgmd*)  DEFAULT_NNODES=4;  DEFAULT_DEBUG=1 ;;
    *)        DEFAULT_NNODES=16; DEFAULT_DEBUG=0 ;;
esac

echo "[submit_bspo_md] combo=$COMBO_ID variant=$VAR delta=$DELTA lambda=$LAMBDA"
echo "[submit_bspo_md] overrides: $ALL_OVR"

NNODES="${NNODES:-$DEFAULT_NNODES}" \
BSPO_DEBUG="${BSPO_DEBUG:-$DEFAULT_DEBUG}" \
    exec "$PROD_SUBMIT" "$COMBO_ID" "$ALL_OVR"
