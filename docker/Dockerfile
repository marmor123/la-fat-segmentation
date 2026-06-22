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
    pip install --no-cache-dir -e .

# ── Workspace directories ───────────────────────────────────────────────────
RUN mkdir -p /workspace/data/raw /workspace/data/intermediate /workspace/outputs

# ── Pre-download TotalSegmentator model weights ─────────────────────────────
# Run a minimal TS invocation to trigger model weight download into
# $TOTALSEG_HOME_DIR/nnunet/.  This bakes ~2-3 GB of pretrained weights
# into the image so users never wait for first-run downloads.
RUN mkdir -p /totalsegmentator && \
    python -c "\
import os, sys; \
os.environ['TOTALSEG_HOME_DIR'] = '/totalsegmentator'; \
print('[build] Downloading TotalSegmentator model weights...'); \
try: \
    from totalsegmentator.lib.download import download_pretrained_weights; \
    download_pretrained_weights(); \
    print('[build] Model weights downloaded successfully.'); \
except Exception as e: \
    print(f'[build] Note: Model download not completed ({e}) — will download on first run.'); \
" 2>&1

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Default command ─────────────────────────────────────────────────────────
ENTRYPOINT ["/entrypoint.sh"]
CMD ["pipeline"]
