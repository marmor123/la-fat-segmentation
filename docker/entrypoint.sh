#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LA Fat Segmentation — Docker Entrypoint
# ---------------------------------------------------------------------------
# 1. Writes the TotalSegmentator license config.
# 2. Dispatches to pipeline (batch) or dashboard mode.
# ---------------------------------------------------------------------------
set -euo pipefail

TOTALSEG_HOME="${TOTALSEG_HOME_DIR:-/totalsegmentator}"
TOTALSEG_CONFIG="${TOTALSEG_HOME}/config.json"

# ── License config ──────────────────────────────────────────────────────────
mkdir -p "${TOTALSEG_HOME}"

if [ -n "${TOTALSEG_LICENSE:-}" ]; then
    cat > "${TOTALSEG_CONFIG}" <<EOF
{
    "totalseg_id": "${TOTALSEG_LICENSE}",
    "send_usage_stats": false,
    "license_number": "${TOTALSEG_LICENSE}",
    "statistics_disclaimer_shown": true
}
EOF
    echo "[entrypoint] License config written to ${TOTALSEG_CONFIG}"
else
    cat > "${TOTALSEG_CONFIG}" <<EOF
{
    "totalseg_id": "",
    "send_usage_stats": false,
    "statistics_disclaimer_shown": true
}
EOF
    echo "[entrypoint] WARNING: No TOTALSEG_LICENSE provided — config written without license"
fi

# ── Resolve data / output directories ───────────────────────────────────────
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
WORKSPACE="${WORKSPACE:-/workspace}"

# Ensure workspace directories exist
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/intermediate"
mkdir -p "${OUTPUT_DIR}"

# ── Dispatch ────────────────────────────────────────────────────────────────
MODE="${1:-pipeline}"

case "${MODE}" in
    pipeline)
        echo "[entrypoint] Starting batch pipeline"
        echo "[entrypoint] Data directory:  ${DATA_DIR}"
        echo "[entrypoint] Output directory: ${OUTPUT_DIR}"
        exec python -m la_fat.batch_pipeline \
            --data-dir "${DATA_DIR}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    dashboard)
        echo "[entrypoint] Serving QA Studio dashboard on port 5006"
        echo "[entrypoint] Output directory: ${OUTPUT_DIR}"
        exec python -m http.server 5006 --directory "${OUTPUT_DIR}" --bind 0.0.0.0
        ;;
    *)
        echo "[entrypoint] ERROR: Unknown mode '${MODE}'. Valid modes: pipeline, dashboard" >&2
        exit 1
        ;;
esac
