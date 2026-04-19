#!/usr/bin/env bash
# bisimpo/submit_bspo_md.sh — launch a BSPO multi-domain RayJob.
#
# Resolves combo_id → {variant, delta, λ_tj, λ_grp, λ_d} for bspo_md,
# then delegates to iter_kuberay_32nodes_verl_training/submit/submit_md.sh.
#
# Usage:
#   NNODES=8 ./submit_bspo_md.sh cbdg-md-v1-smoke       # debug smoke (stage 1 uses loss_mode=bspo)
#   NNODES=4 ./submit_bspo_md.sh cbmd-v1                 # formal run (V1 simplest, 120 steps)
#
# NNODES defaults to 8 for cbdg-md-* combos, 4 for cbmd-* combos.
#
# Stage-1 hack: for cbdg-md-*-smoke combos we intentionally submit with the
# stock single-domain `bspo` loss (NOT `bspo_md`). Stage-1 goal is to verify
# the infra pipe (kuberay → Gym → Hydra compose → data loader → val cycle)
# with a proven loss path before introducing the new multi-domain code.
# Stage 2+ (post-plumbing) will flip cbdg-md-* over to loss_mode=bspo_md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_SUBMIT="$HERE/../iter_kuberay_32nodes_verl_training/submit/submit_md.sh"

COMBO_ID="${1:?usage: $0 <combo_id> [extra Hydra overrides]}"
shift
EXTRA_USER="${*:-}"

# ── Per-combo hparams ─────────────────────────────────────────────────────────
# Variant → loss_mode mapping happens implicitly:
#   Stage 1 smokes (cbdg-md-*-smoke): loss_mode=bspo (sd), variant from combo → simplest/w_penalty/w_penalty_only (only sd variants available)
#                                     V2 hierarchy and V3 strict_wasserstein SKIPPED in stage 1 (they don't exist in sd)
#   Stage 2+:                         loss_mode=bspo_md, all 5 variants available.
# Default stage is "1" (infra). Export STAGE=2 or STAGE=3 to flip loss_mode.
STAGE="${STAGE:-1}"

case "$COMBO_ID" in
    # ── Debug smokes ────────────────────────────────────────────────────────
    cbdg-md-v1-smoke)  VAR=simplest;       DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbdg-md-v2-smoke)  VAR=hierarchy;      DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbdg-md-v3-smoke)  VAR=strict_wasserstein; DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbdg-md-v4-smoke)  VAR=w_penalty;      DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbdg-md-v5-smoke)  VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    # ── Formal 120-step runs ────────────────────────────────────────────────
    cbmd-v1)           VAR=simplest;       DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbmd-v2)           VAR=hierarchy;      DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbmd-v3)           VAR=strict_wasserstein; DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbmd-v4)           VAR=w_penalty;      DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    cbmd-v5)           VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA_TJ=1.0e-2; LAMBDA_GRP=1.0e-2; LAMBDA_D=1.0 ;;
    *) echo "error: unknown md combo '$COMBO_ID'" >&2; exit 2 ;;
esac

# ── NNODES default (8 for cbdg-md-*, 4 for cbmd-*) ────────────────────────────
if [[ "$COMBO_ID" == cbdg-md-* ]]; then
    NNODES="${NNODES:-8}"
elif [[ "$COMBO_ID" == cbmd-v2 ]] || [[ "$COMBO_ID" == cbmd-v3 ]]; then
    # V2/V3 benefit from 8-node throughput (2× faster wall-clock per step).
    NNODES="${NNODES:-8}"
elif [[ "$COMBO_ID" == cbmd-v1 ]]; then
    # V1 baseline also on 8-node for fair comparison with V2/V3.
    NNODES="${NNODES:-8}"
else
    NNODES="${NNODES:-4}"
fi

# ── Loss mode by stage ────────────────────────────────────────────────────────
case "$STAGE" in
    1)  LOSS_MODE=bspo ;;        # stage 1 uses the proven sd loss for infra validation
    2|3) LOSS_MODE=bspo_md ;;    # stage 2+ uses the new multi-domain loss
    *)  echo "error: invalid STAGE=$STAGE (use 1, 2, or 3)" >&2; exit 2 ;;
esac

# Stage-1 guard: V2 and V3 don't exist in single-domain bspo. Force-map them to simplest.
if [ "$LOSS_MODE" = "bspo" ] && { [ "$VAR" = "hierarchy" ] || [ "$VAR" = "strict_wasserstein" ]; }; then
    echo "[submit_bspo_md] stage=1 (loss=bspo): variant=$VAR not available in sd bspo — mapping to 'simplest' for infra smoke"
    VAR=simplest
fi

# ── BSPO override string (yaml → CLI; stage 1 omits λ_grp/λ_d which don't exist yet) ───
BSPO_OVR=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE"
    "actor_rollout_ref.actor.bspo_variant=$VAR"
    "actor_rollout_ref.actor.bspo_delta=$DELTA"
    "actor_rollout_ref.actor.bspo_lambda_tj=$LAMBDA_TJ"
)
if [ "$LOSS_MODE" = "bspo_md" ]; then
    # These config fields only exist after Stage 3 adds them to actor.yaml.
    BSPO_OVR+=(
        "actor_rollout_ref.actor.bspo_lambda_grp=$LAMBDA_GRP"
        "actor_rollout_ref.actor.bspo_lambda_d=$LAMBDA_D"
    )
fi

ALL_OVR="${BSPO_OVR[*]} ${EXTRA_USER}"

echo "[submit_bspo_md] combo=$COMBO_ID stage=$STAGE loss=$LOSS_MODE variant=$VAR"
echo "[submit_bspo_md] δ=$DELTA λ_tj=$LAMBDA_TJ λ_grp=$LAMBDA_GRP λ_d=$LAMBDA_D nnodes=$NNODES"
echo "[submit_bspo_md] overrides: $ALL_OVR"

NNODES="$NNODES" exec "$PROD_SUBMIT" "$COMBO_ID" "$ALL_OVR"
