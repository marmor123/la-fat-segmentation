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
# Write a temporary license config so gated tasks (heartchambers_highres,
# trunk_cavities) can download.  Then download weights for all three TS
# tasks used by the pipeline.  Models land in $TOTALSEG_HOME_DIR/nnunet/.
# This bakes ~2-3 GB of pretrained weights into the image.
RUN mkdir -p /totalsegmentator && \
    echo "{\"totalseg_id\":\"${TOTALSEG_LICENSE}\",\"license_number\":\"${TOTALSEG_LICENSE}\",\"send_usage_stats\":false,\"statistics_disclaimer_shown\":true}" > /totalsegmentator/config.json && \
    python -c "\
import os; \
os.environ['TOTALSEG_HOME_DIR'] = '/totalsegmentator'; \
from totalsegmentator.libs import download_pretrained_weights; \
from totalsegmentator.config import setup_totalseg, set_config_key; \
setup_totalseg(); \
set_config_key('statistics_disclaimer_shown', True); \
tasks = {'heartchambers_highres': [301], 'total': [291,292,293,294,295,298], 'trunk_cavities': [343]}; \
for name, ids in tasks.items(): \
    for tid in ids: \
        print(f'[build] Downloading {name} (task {tid})...'); \
        download_pretrained_weights(tid); \
print('[build] All model weights downloaded.'); \
"

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Default command ─────────────────────────────────────────────────────────
ENTRYPOINT ["/entrypoint.sh"]
CMD ["pipeline"]
