#!/usr/bin/env bash
# bisimpo/submit_bspo.sh — launch a BSPO RayJob.
#
# Resolves BSPO-specific Hydra overrides from combo_id → {variant, delta, lambda},
# then delegates to iter_kuberay_32nodes_verl_training/submit/submit.sh.
#
# Usage:
#   ./submit_bspo.sh <combo_id> [<extra Hydra overrides>...]
#
# Defaults applied on top of each run:
#   loss_mode=bspo
#   test_freq=10
#   val_before_train=true
#   val_max_samples=128
#   save_freq=9999  (no checkpoint writes — metrics are enough for ablation)
#
# NNODES env var selects the compute profile (see iter_kuberay_.../submit/submit.sh).
# Defaults to 4 (C11: no-offload, 4n, gmu=0.4 — maximum cluster throughput).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_SUBMIT="$HERE/../iter_kuberay_32nodes_verl_training/submit/submit.sh"

COMBO_ID="${1:?usage: $0 <combo_id> [extra Hydra overrides]}"
shift
EXTRA_USER="${*:-}"

# ── Per-combo BSPO hparams (Phase-1 defaults; Phase-2 sweeps override via CLI) ─
# LR_WARMUP must be strictly less than total_training_steps (Megatron assert in
# optimizer_param_scheduler.py). For 1-step smoke we use 0; for 120-step runs, 10.
case "$COMBO_ID" in
    cdbgbspo)      VAR=simplest;       DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=0 ;;
    cbsp101)       VAR=simplest;       DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbsp103)       VAR=w_penalty;      DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    cbsp104)       VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;;
    # ── Phase-2 preview slots (single-domain hparam variants) ─────────────
    # Used to fill idle GPU capacity while Phase-1 v2 is in flight. Picks
    # target the specific failure modes observed in Phase-1 progress logs.
    cbsp201)       VAR=simplest;       DELTA=1.0e-3; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # V1 looser clip (V1@3e-4 saturated)
    cbsp202)       VAR=simplest;       DELTA=1.0e-2; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # V1 much looser
    cbsp203|cbsp204|cbsp205) VAR=simplest; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # reserved slots
    cbsp301)       VAR=w_penalty;      DELTA=1.0e-3; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # V4 looser clip
    cbsp302)       VAR=w_penalty;      DELTA=3.0e-4; LAMBDA=1.0e-2; LR_WARMUP=10 ;; # V4 stronger penalty (addresses s_tau drift)
    cbsp303|cbsp304|cbsp305|cbsp306|cbsp307|cbsp308|cbsp309)
                    VAR=w_penalty;      DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # reserved slots
    cbsp401)       VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA=1.0e-2; LR_WARMUP=10 ;; # V5 stronger penalty
    cbsp402|cbsp403|cbsp404|cbsp405|cbsp406|cbsp407|cbsp408|cbsp409)
                    VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA=1.0e-3; LR_WARMUP=10 ;; # reserved slots
    *)  echo "error: unknown BSPO combo '$COMBO_ID'" >&2; exit 2 ;;
esac

# ── BSPO override string (yaml → CLI; preferred over bash env vars) ───────────
BSPO_OVR=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=bspo"
    "actor_rollout_ref.actor.bspo_variant=$VAR"
    "actor_rollout_ref.actor.bspo_delta=$DELTA"
    "actor_rollout_ref.actor.bspo_lambda_tj=$LAMBDA"
)

# ── Ablation knobs (non-Category-B, safe to set at run level) ────────────────
# val_max_samples is inherited from 40bra_16node_sd.yaml (-1 = all 230 rows of
# the hard-eval parquet). Overriding it here would truncate the eval set
# non-deterministically across data_sources, so we leave it.
ABLATE_OVR=(
    "trainer.test_freq=10"
    "trainer.val_before_train=true"
    "trainer.save_freq=9999"
    "trainer.log_val_generations=0"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP"
)

ALL_OVR="${BSPO_OVR[*]} ${ABLATE_OVR[*]} ${EXTRA_USER}"

echo "[submit_bspo] combo=$COMBO_ID variant=$VAR delta=$DELTA lambda=$LAMBDA"
echo "[submit_bspo] overrides: $ALL_OVR"

# Delegate — preserving NNODES env var from caller (default 4 below if unset)
NNODES="${NNODES:-4}" exec "$PROD_SUBMIT" "$COMBO_ID" "$ALL_OVR"
