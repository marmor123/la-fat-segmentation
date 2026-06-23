# ---------------------------------------------------------------------------
# LA Fat Segmentation — Docker Image
# ---------------------------------------------------------------------------
# Build:
#   docker build --build-arg TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX -t la-fat:latest .
#
# The image auto-detects GPU availability via PyTorch and falls back to CPU.
# ---------------------------------------------------------------------------

FROM pytorch/pytorch:latest

LABEL maintainer="marmor123"
LABEL description="LA Fat Segmentation — epicardial adipose tissue quantification from CT scans"

# ── Build arguments ─────────────────────────────────────────────────────────
ARG TOTALSEG_LICENSE=""

# ── Environment ─────────────────────────────────────────────────────────────
ENV TOTALSEG_HOME_DIR=/totalsegmentator
ENV TOTALSEG_LICENSE=${TOTALSEG_LICENSE}
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        && rm -rf /var/lib/apt/lists/*

# ── Clone & install pipeline ────────────────────────────────────────────────
WORKDIR /app
RUN git clone https://github.com/marmor123/la-fat-segmentation.git . && \
    pip install --no-cache-dir .

# ── Workspace directories ───────────────────────────────────────────────────
RUN mkdir -p /workspace/data/raw /workspace/data/intermediate /workspace/outputs

# ── Pre-download TotalSegmentator model weights ─────────────────────────────
# Copy the download script, write a temporary license config so gated tasks
# can download, then fetch all three TS model sets.
# Models land in $TOTALSEG_HOME_DIR/nnunet/ (~2-3 GB).
COPY docker/download_weights.py /tmp/download_weights.py
RUN mkdir -p /totalsegmentator && \
    printf '{"totalseg_id":"%s","license_number":"%s","send_usage_stats":false,"statistics_disclaimer_shown":true}' \
        "${TOTALSEG_LICENSE}" "${TOTALSEG_LICENSE}" > /totalsegmentator/config.json && \
    python /tmp/download_weights.py && \
    rm /tmp/download_weights.py

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Default command ─────────────────────────────────────────────────────────
ENTRYPOINT ["/entrypoint.sh"]
CMD ["pipeline"]
