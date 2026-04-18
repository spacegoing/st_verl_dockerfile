#!/usr/bin/env bash
# 0418/pipeline.sh — end-to-end debug-run orchestration.
#
# Waits for one RayJob to reach terminal state, pulls its perf_log.jsonl,
# runs analyze_perf.py, writes a summary, and optionally launches the next
# profile.
#
# Usage:
#   ./pipeline.sh <rayjob_name> <profile_label> [next_profile_nnodes]
#
# Examples:
#   ./pipeline.sh bra40-dbg-cdbg5-16n-2mvbt P0-16node 8      # wait → analyze → submit 8-node
#   ./pipeline.sh bra40-dbg-cdbg5-8n-abcde  P1-8node  4      # wait → analyze → submit 4-node
#   ./pipeline.sh bra40-dbg-cdbg5-4n-xxxxx  P2-4node         # wait → analyze → stop

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export NO_PROXY="${K8S_API_IP:-180.184.249.201}" \
       no_proxy="${K8S_API_IP:-180.184.249.201}"

JOB="${1:?usage: $0 <rayjob_name> <profile_label> [next_profile_nnodes]}"
LABEL="${2:?usage: $0 <rayjob_name> <profile_label> [next_profile_nnodes]}"
NEXT_N="${3:-}"

PIPELINE_LOG=/tmp/0418_pipeline_${LABEL}.log
SUMMARY_DIR=/mnt/public/lichang93/st_verl_dockerfile/0418/results
mkdir -p "$SUMMARY_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PIPELINE_LOG"; }

log "pipeline started  job=$JOB  label=$LABEL  next_nnodes=${NEXT_N:-none}"

# ── Phase 1 — wait for terminal state ────────────────────────────────────────
POLL=120   # 2 min
TIMEOUT=$((90*60))   # 90 min hard cap per profile
ELAPSED=0
LAST_STATUS=""

while true; do
    STATUS=$(kubectl get rayjob "$JOB" -o jsonpath='{.status.jobStatus}' 2>/dev/null || echo "MISSING")
    DEPLOY=$(kubectl get rayjob "$JOB" -o jsonpath='{.status.jobDeploymentStatus}' 2>/dev/null || echo "?")
    if [ "$STATUS" != "$LAST_STATUS" ]; then
        log "status change: $LAST_STATUS → $STATUS  deploy=$DEPLOY"
        LAST_STATUS=$STATUS
    fi
    case "$STATUS" in
        SUCCEEDED|FAILED|STOPPED)
            log "job reached terminal state: $STATUS"
            break
            ;;
        MISSING)
            log "RayJob disappeared; aborting"
            exit 2
            ;;
    esac
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "timeout after ${TIMEOUT}s; stopping monitor"
        break
    fi
    sleep "$POLL"
    ELAPSED=$((ELAPSED+POLL))
done

# ── Phase 2 — collect perf_log.jsonl and training log ────────────────────────
# Problem we fixed 2026-04-18:
#   A naive `ls -dt .../40bra_k8s_16node_sd_${COMBO}_*` picks the newest ckpt
#   dir matching the combo. If the current run failed before writing any
#   perf_log, that glob returns a PREVIOUS successful run's dir and we then
#   report stale data as the current result. Worse, Phase 5 deleted the
#   current failed RayJob so the OOM pods were lost.
# Fix:
#   Treat a perf_log as valid for THIS run only if its first record's
#   wall_clock_utc is AFTER the RayJob's creationTimestamp (minus a small
#   skew for pod-init lag). Otherwise the log belongs to an older run.
#   Always also copy the PFS training log from verl/logs/... as a fallback.

CKPT_ROOT=/mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain
LEGACY_LOG_ROOT=/mnt/public/lichang93/st_verl_dockerfile/verl/logs/40bra_k8s_single_domain
LEGACY_LOOKUP=$CKPT_ROOT/.rayjob_lookup

ENTRYPOINT=$(kubectl get rayjob "$JOB" -o jsonpath='{.spec.entrypoint}' 2>/dev/null || true)
CREATED_AT=$(kubectl get rayjob "$JOB" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || true)
COMBO=$(echo "$ENTRYPOINT" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^c[a-z]*[0-9]+$/) {print $i; exit}}')

if [ -n "$CREATED_AT" ]; then
    CREATE_EPOCH=$(date -u -d "$CREATED_AT" +%s 2>/dev/null || echo 0)
else
    CREATE_EPOCH=0
fi
CUTOFF_EPOCH=$((CREATE_EPOCH - 300))
log "rayjob combo=$COMBO created=$CREATED_AT"

CKPT_DIR=""

# Primary (post-2026-04-18 layout): run dir IS the RayJob name. Same dir contains
# run.log, perf_log.jsonl, exp_name.txt, label.txt. No lookup indirection needed.
if [ -d "$CKPT_ROOT/$JOB" ]; then
    CKPT_DIR="$CKPT_ROOT/$JOB"
    log "run dir (direct by rayjob name) → $CKPT_DIR"

# Fallback 1 (transitional): marker file from the brief .rayjob_lookup era.
elif [ -r "$LEGACY_LOOKUP/$JOB" ]; then
    exp=$(cat "$LEGACY_LOOKUP/$JOB" 2>/dev/null)
    if [ -n "$exp" ] && [ -d "$CKPT_ROOT/$exp" ]; then
        CKPT_DIR="$CKPT_ROOT/$exp"
        log "run dir (legacy marker) → $CKPT_DIR"
    fi
fi

# Fallback 2 (transitional): time-bounded search for the oldest layout where
# the dir name was the full exp_name. Accept a dir only if its perf_log's
# first record was written after the RayJob was submitted.
if [ -z "$CKPT_DIR" ]; then
    for cand in $(ls -dt "$CKPT_ROOT"/40bra_k8s_16node_sd_${COMBO}_* 2>/dev/null); do
        f="$cand/perf_log.jsonl"
        [ -r "$f" ] || continue
        first_ts=$(python3 -c "import json,sys,datetime;
try:
    line=open('$f').readline().strip()
    if not line: sys.exit()
    ts=json.loads(line).get('wall_clock_utc','')
    if ts.endswith('Z'): ts=ts[:-1]
    print(int(datetime.datetime.fromisoformat(ts).timestamp()))
except Exception: pass" 2>/dev/null)
        [ -n "$first_ts" ] || continue
        if [ "$first_ts" -ge "$CUTOFF_EPOCH" ]; then
            CKPT_DIR="$cand"
            log "run dir (legacy time-match fallback) → $cand"
            break
        fi
    done
fi

PERF_FILE=""
[ -n "$CKPT_DIR" ] && [ -r "$CKPT_DIR/perf_log.jsonl" ] && PERF_FILE="$CKPT_DIR/perf_log.jsonl"

# Copy perf_log to the flat results/ index for quick comparison; the canonical
# copy stays inside the run dir.
if [ -n "$PERF_FILE" ]; then
    cp "$PERF_FILE" "$SUMMARY_DIR/${LABEL}_perf_log.jsonl"
    log "perf_log → $SUMMARY_DIR/${LABEL}_perf_log.jsonl  (canonical: $PERF_FILE)"
else
    log "WARN: no perf_log for this run (likely failed before first training step)"
fi

# Training log. New layout keeps run.log inside the run dir. Old layout kept it
# in verl/logs/... with a filename starting with the combo + timestamp.
TRAIN_LOG=""
if [ -n "$CKPT_DIR" ] && [ -r "$CKPT_DIR/run.log" ]; then
    TRAIN_LOG="$CKPT_DIR/run.log"
else
    TRAIN_LOG=$(ls -t "$LEGACY_LOG_ROOT"/40bra_k8s_16node_sd_${COMBO}_*.log 2>/dev/null | while read f; do
        mt=$(stat -c%Y "$f" 2>/dev/null || echo 0)
        [ "$mt" -ge "$CREATE_EPOCH" ] && { echo "$f"; break; }
    done)
fi
if [ -n "$TRAIN_LOG" ] && [ -r "$TRAIN_LOG" ]; then
    cp "$TRAIN_LOG" "$SUMMARY_DIR/${LABEL}_training.log"
    log "training log → $SUMMARY_DIR/${LABEL}_training.log  (canonical: $TRAIN_LOG)"
fi

# ── Phase 3 — analyze ────────────────────────────────────────────────────────
if [ -r "$SUMMARY_DIR/${LABEL}_perf_log.jsonl" ]; then
    python3 "$HERE/analyze_perf.py" "$SUMMARY_DIR/${LABEL}_perf_log.jsonl" \
        > "$SUMMARY_DIR/${LABEL}_analysis.txt" 2>&1 || true
    log "analysis written to $SUMMARY_DIR/${LABEL}_analysis.txt"
    tail -5 "$SUMMARY_DIR/${LABEL}_analysis.txt" | tee -a "$PIPELINE_LOG"
    # Also put the analysis alongside the run's other files on PFS
    if [ -n "$CKPT_DIR" ] && [ -w "$CKPT_DIR" ]; then
        cp "$SUMMARY_DIR/${LABEL}_analysis.txt" "$CKPT_DIR/analysis.txt" 2>&1 \
            | head -1 | while read line; do log "analysis copy: $line"; done || true
    fi
fi

# ── Phase 4 — submit next profile if requested ───────────────────────────────
# Natural experiment order: 16 → 8 → 4. When called with NEXT_N=8 we
# submit an 8-node job and chain a follow-up with NEXT_N=4 so the whole
# P0→P1→P2 ladder runs unattended.
next_after() {
    case "$1" in
        8) echo 4 ;;
        4) echo "" ;;
        *) echo "" ;;
    esac
}

if [ -n "$NEXT_N" ] && [ "$LAST_STATUS" = "SUCCEEDED" ]; then
    log "submitting next profile: $NEXT_N nodes"
    NEXT_OUT=$("$HERE/submit_debug.sh" cdbg5 "$NEXT_N" 'data.val_max_samples=32 trainer.test_freq=3' 2>&1)
    NEXT_JOB=$(echo "$NEXT_OUT" | grep -oE 'bra40-dbg-cdbg5-[0-9]+n-[a-z0-9]+' | head -1)
    log "next job: $NEXT_JOB"
    if [ -n "$NEXT_JOB" ]; then
        NEXT_LABEL="P-${NEXT_N}node"
        NEXT_NEXT_N=$(next_after "$NEXT_N")
        nohup bash "$HERE/pipeline.sh" "$NEXT_JOB" "$NEXT_LABEL" $NEXT_NEXT_N > /dev/null 2>&1 &
        log "chained pipeline PID=$! for $NEXT_JOB (next_after=${NEXT_NEXT_N:-none})"
    fi
elif [ -n "$NEXT_N" ]; then
    log "NOT submitting next profile because last status was $LAST_STATUS (not SUCCEEDED)"
fi

# ── Phase 5 — explicit RayJob delete to free nodes immediately ───────────────
# The RayJob's ttlSecondsAfterFinished would eventually clean up anyway, but
# deleting now returns nodes to the Volcano queue as soon as we've collected
# results — critical when many C-runs are queued behind this one.
if [ -r "$SUMMARY_DIR/${LABEL}_perf_log.jsonl" ]; then
    kubectl delete rayjob "$JOB" --wait=false 2>&1 | head -3 | while read line; do log "delete: $line"; done
else
    log "SKIP explicit delete — perf_log missing; letting ttl clean up so head pod logs stay readable"
fi

log "pipeline done for $LABEL"
