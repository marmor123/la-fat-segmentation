#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LA Fat Segmentation — Rebuild & Export
# ---------------------------------------------------------------------------
# Maintainer script: builds the Docker image and exports it as a tar file
# for offline distribution.  Requires the TOTALSEG_LICENSE environment variable
# to be set (or pass it inline).
#
# Usage:
#   export TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX
#   bash docker/rebuild.sh
#
#   Or inline:
#   TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX bash docker/rebuild.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE_NAME="${IMAGE_NAME:-la-fat}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TAR_NAME="${TAR_NAME:-la-fat-image.tar}"

LICENSE="${TOTALSEG_LICENSE:-}"

if [[ -z "${LICENSE}" ]]; then
    echo "============================================================"
    echo "  WARNING: TOTALSEG_LICENSE is not set."
    echo "  The image will build but TotalSegmentator will not have"
    echo "  a valid license for gated tasks (heartchambers_highres,"
    echo "  trunk_cavities)."
    echo "============================================================"
    echo ""
    echo "Set it with:"
    echo "  export TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX"
    echo ""
    read -rp "Continue without license? [y/N] " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "============================================================"
echo "  BUILDING DOCKER IMAGE"
echo "============================================================"
echo ""
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Tar:   ${TAR_NAME}"
echo ""

docker build \
    --build-arg TOTALSEG_LICENSE="${LICENSE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f docker/Dockerfile \
    .

echo ""
echo "============================================================"
echo "  EXPORTING TO TAR"
echo "============================================================"
echo ""

docker save -o "${TAR_NAME}" "${IMAGE_NAME}:${IMAGE_TAG}"

TAR_SIZE=$(du -h "${TAR_NAME}" | cut -f1)
echo ""
echo "  Image saved: ${TAR_NAME} (${TAR_SIZE})"
echo ""
echo "============================================================"
echo "  DISTRIBUTION PACKAGE READY"
echo "============================================================"
echo ""
echo "  Copy these files to a USB drive or network share:"
echo ""
echo "    la-fat-image.tar"
echo "    docker/Install.bat"
echo "    docker/install.sh"
echo "    docker/Process Scans.bat"
echo "    docker/View Results.bat"
echo "    docker/Process Scans.desktop"
echo "    docker/View Results.desktop"
echo ""
