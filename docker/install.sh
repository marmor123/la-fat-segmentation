#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LA Fat Segmentation — Linux Installer
# ---------------------------------------------------------------------------
# Run this script to install.  Requires Docker.
# Re-running is safe — your data folder is never deleted.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "============================================================"
echo "  LA FAT SEGMENTATION — INSTALLER"
echo "============================================================"
echo ""

# --- Check for Docker -------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "[ERROR] Docker is not installed or not on PATH."
    echo "Please install Docker from https://docs.docker.com/engine/install/"
    echo ""
    exit 1
fi
echo "[OK] Docker found."

# --- Load image --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAR="${SCRIPT_DIR}/la-fat-image.tar"

if [[ ! -f "${IMAGE_TAR}" ]]; then
    echo "[ERROR] Image file not found: ${IMAGE_TAR}"
    echo "Please ensure la-fat-image.tar is in the same folder as this script."
    exit 1
fi

echo "Loading Docker image..."
docker load -i "${IMAGE_TAR}"
echo "[OK] Image loaded."

# --- Create data folder on Desktop -------------------------------------------
DATA_DIR="${HOME}/Desktop/la-fat-data"

if [[ ! -d "${DATA_DIR}" ]]; then
    mkdir -p "${DATA_DIR}/data/raw"
    mkdir -p "${DATA_DIR}/data/intermediate"
    mkdir -p "${DATA_DIR}/outputs"
    echo "[OK] Created data folder: ${DATA_DIR}"
else
    echo "[OK] Data folder already exists: ${DATA_DIR}"
fi

# --- Copy desktop shortcuts --------------------------------------------------
cp "${SCRIPT_DIR}/Process Scans.desktop" "${HOME}/Desktop/Process Scans.desktop"
cp "${SCRIPT_DIR}/View Results.desktop"   "${HOME}/Desktop/View Results.desktop"

# Make desktop files executable / trusted
chmod +x "${HOME}/Desktop/Process Scans.desktop" 2>/dev/null || true
chmod +x "${HOME}/Desktop/View Results.desktop"   2>/dev/null || true

# Mark as trusted so GNOME/KDE doesn't show security warnings
gio set "${HOME}/Desktop/Process Scans.desktop" metadata::trusted true 2>/dev/null || true
gio set "${HOME}/Desktop/View Results.desktop"   metadata::trusted true 2>/dev/null || true

echo "[OK] Desktop shortcuts created."

echo ""
echo "============================================================"
echo "  INSTALL COMPLETE"
echo "============================================================"
echo ""
echo "  Data folder:  ${DATA_DIR}"
echo "  Desktop shortcuts:"
echo "    - Process Scans  (double-click to process CT scans)"
echo "    - View Results   (double-click to view dashboard)"
echo ""
echo "  Usage:"
echo "    1. Drop .nii.gz CT scans into: ${DATA_DIR}/data/raw/"
echo "    2. Double-click 'Process Scans' on your Desktop"
echo "    3. Double-click 'View Results' to view QA dashboards"
echo ""
