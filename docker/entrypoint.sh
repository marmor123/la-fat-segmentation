#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LA Fat Segmentation — Docker Entrypoint
# ---------------------------------------------------------------------------
# 1. Writes the TotalSegmentator license config if provided.
# 2. Dispatches to `la-fat` CLI or serves QA Studio.
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
fi

# ── Resolve data / output directories ───────────────────────────────────────
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"

mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/intermediate"
mkdir -p "${OUTPUT_DIR}"

MODE="${1:-batch}"

case "${MODE}" in
    pipeline|batch)
        echo "[entrypoint] Starting LA Fat batch pipeline"
        echo "[entrypoint] Data directory:   ${DATA_DIR}"
        echo "[entrypoint] Output directory: ${OUTPUT_DIR}"
        exec la-fat batch \
            --data-dir "${DATA_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --no-open
        ;;
    dashboard)
        echo "[entrypoint] Serving QA Studio on port 8080"
        echo "[entrypoint] Output directory: ${OUTPUT_DIR}"
        exec python -m http.server 8080 --directory "${OUTPUT_DIR}" --bind 0.0.0.0
        ;;
    check)
        exec la-fat check
        ;;
    *)
        exec la-fat "$@"
        ;;
esac
