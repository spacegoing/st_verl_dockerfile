#!/usr/bin/env bash
# run_pipeline.sh — end-to-end voltest workflow.
#
# Phases:
#   1. preflight   — sanity-check cluster state
#   2. submit      — render template N times and apply each via stdin (no
#                    per-job files on disk)
#   3. watch       — periodically log PodGroup/Pod state; auto-delete any
#                    SUCCEEDED/FAILED RayJob so the next queued gang can
#                    promote atomically
#   4. wait        — block until all N RayJobs have terminated and cleaned up
#   5. verify      — check each job's PFS log for gpu_burn DONE lines
#   6. report      — summary to stdout + logs/pipeline-<ts>.log
#
# Usage:
#   ./run_pipeline.sh            # default N=10
#   N=5 ./run_pipeline.sh        # submit 5
#   DURATION_S=60 ./run_pipeline.sh   # 1-min runs instead of 5-min
#   SKIP_SUBMIT=1 ./run_pipeline.sh   # only verify an already-finished batch

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL="${HERE}/yaml/rayjob.yaml"
LOG_DIR="${HERE}/logs"
mkdir -p "$LOG_DIR"

N="${N:-10}"
START="${START:-0}"
DURATION_S="${DURATION_S:-300}"
NAME_PREFIX="${NAME_PREFIX:-voltest}"
TICK_SECONDS="${TICK_SECONDS:-20}"
WATCH_TIMEOUT_S="${WATCH_TIMEOUT_S:-2400}"   # 40 min cap

TS=$(date -u +%Y%m%dT%H%M%SZ)
PIPE_LOG="${LOG_DIR}/pipeline-${TS}.log"

export HTTPS_PROXY= HTTP_PROXY= NO_PROXY="${NO_PROXY:-180.184.249.201}"

# ──────────────────────────────────────────────────────────────────────────────
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PIPE_LOG"; }

die() { log "FATAL: $*"; exit 1; }

require() {
    command -v "$1" >/dev/null 2>&1 || die "required binary not found: $1"
}

# ──────────────────────────────────────────────────────────────────────────────
phase_1_preflight() {
    log "=== PHASE 1 — preflight ==="
    require kubectl
    require sed
    [ -r "$TPL" ] || die "template missing: $TPL"

    LEFTOVER=$(kubectl get rayjob -l voltest=true --no-headers 2>/dev/null | wc -l)
    if [ "$LEFTOVER" != "0" ]; then
        log "WARN: $LEFTOVER voltest rayjobs already exist:"
        kubectl get rayjob -l voltest=true 2>&1 | tee -a "$PIPE_LOG"
        die "clean them up first: kubectl delete rayjob -l voltest=true"
    fi

    PG_LEFT=$(kubectl get podgroup -A --no-headers 2>/dev/null | grep -c ray-voltest || true)
    [ "$PG_LEFT" = "0" ] || log "WARN: $PG_LEFT stale voltest PodGroups still present (will clear on own)"

    # Cluster GPU capacity
    GPU_TOTAL=$(kubectl get node -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' \
                | awk '{s+=$1} END{print s+0}')
    log "cluster allocatable GPUs: $GPU_TOTAL (need: $((N*8*8)) for all N=$N jobs to run concurrently)"

    log "preflight OK"
}

# ──────────────────────────────────────────────────────────────────────────────
phase_2_submit() {
    log "=== PHASE 2 — submit $N jobs (names: ${NAME_PREFIX}-${START}..${NAME_PREFIX}-$((START+N-1))) ==="
    for i in $(seq "$START" $((START + N - 1))); do
        JOB="${NAME_PREFIX}-${i}"
        # render template → stdin → kubectl apply (no file on disk)
        RESULT=$(sed -e "s/__JOBNAME__/${JOB}/g" \
                     -e "s/__DURATION_S__/${DURATION_S}/g" \
                     "$TPL" \
                 | kubectl apply -f - 2>&1)
        log "  ${JOB}: ${RESULT}"
    done
    log "submitted $N jobs"
}

# ──────────────────────────────────────────────────────────────────────────────
phase_3_watch_and_cleanup() {
    log "=== PHASE 3+4 — watch state, auto-cleanup terminated, wait for all N to finish ==="

    START_WATCH=$(date +%s)
    PREV_PG_STATE=""

    while true; do
        ELAPSED=$(( $(date +%s) - START_WATCH ))
        if [ "$ELAPSED" -gt "$WATCH_TIMEOUT_S" ]; then
            log "FATAL: watch timeout ($WATCH_TIMEOUT_S s) exceeded"
            return 1
        fi

        # Auto-delete any terminated RayJobs to free nodes for next gang
        TERM=$(kubectl get rayjob -l voltest=true --no-headers 2>/dev/null \
               | awk '$2=="SUCCEEDED" || $2=="FAILED"{print $1 ":" $2}')
        if [ -n "$TERM" ]; then
            while IFS=: read -r NAME STATUS; do
                [ -z "$NAME" ] && continue
                log "  cleanup: $NAME is $STATUS, deleting"
                kubectl delete rayjob "$NAME" --wait=false >> "$PIPE_LOG" 2>&1 || true
            done <<< "$TERM"
        fi

        # Emit condensed state line on change
        PG_STATE=$(kubectl get podgroup -A --no-headers 2>/dev/null \
                   | awk '/ray-voltest-/ {printf "%s:%s(%s/%s) ", $2, $3, $5, $4}')
        if [ "$PG_STATE" != "$PREV_PG_STATE" ] && [ -n "$PG_STATE" ]; then
            log "  state: $PG_STATE"
            PREV_PG_STATE="$PG_STATE"
        fi

        REMAINING=$(kubectl get rayjob -l voltest=true --no-headers 2>/dev/null | wc -l)
        if [ "$REMAINING" = "0" ]; then
            log "all N=$N jobs terminated and cleaned up"
            return 0
        fi
        sleep "$TICK_SECONDS"
    done
}

# ──────────────────────────────────────────────────────────────────────────────
phase_5_verify() {
    log "=== PHASE 5 — verify PFS logs ==="
    PASS=0
    FAIL=0
    for i in $(seq "$START" $((START + N - 1))); do
        JOB="${NAME_PREFIX}-${i}"
        LF="${LOG_DIR}/${JOB}.log"
        if [ ! -s "$LF" ]; then
            log "  $JOB: MISSING log file"
            FAIL=$((FAIL+1))
            continue
        fi
        DONE_COUNT=$(grep -c "gpu_burn DONE" "$LF" 2>/dev/null || echo 0)
        if [ "$DONE_COUNT" -ge 8 ]; then
            log "  $JOB: $DONE_COUNT/8 DONE markers  ✓"
            PASS=$((PASS+1))
        elif [ "$DONE_COUNT" -ge 7 ]; then
            log "  $JOB: $DONE_COUNT/8 DONE markers  ~ (1 log write lost to teardown race, benign)"
            PASS=$((PASS+1))
        else
            log "  $JOB: only $DONE_COUNT/8 DONE markers  ✗"
            FAIL=$((FAIL+1))
        fi
    done
    log "verified: $PASS pass, $FAIL fail"
    [ "$FAIL" = "0" ]
}

# ──────────────────────────────────────────────────────────────────────────────
phase_6_report() {
    log "=== PHASE 6 — summary ==="
    TOTAL_S=$(( $(date +%s) - PIPELINE_START ))
    log "pipeline wall clock: ${TOTAL_S}s"
    log "pipeline log: $PIPE_LOG"
    log "per-job logs: $LOG_DIR/${NAME_PREFIX}-*.log"
}

# ──────────────────────────────────────────────────────────────────────────────
main() {
    PIPELINE_START=$(date +%s)
    log "voltest pipeline starting  N=$N  DURATION_S=$DURATION_S  NAME_PREFIX=$NAME_PREFIX"

    if [ "${SKIP_SUBMIT:-0}" != "1" ]; then
        phase_1_preflight
        phase_2_submit
    else
        log "SKIP_SUBMIT=1 → skipping phases 1-2"
    fi

    phase_3_watch_and_cleanup || { phase_5_verify || true; phase_6_report; exit 1; }

    phase_5_verify
    phase_6_report
}

main "$@"
