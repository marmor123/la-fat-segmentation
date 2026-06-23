"""Download TotalSegmentator pretrained weights during Docker build."""
import os

os.environ["TOTALSEG_HOME_DIR"] = "/totalsegmentator"

from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_totalseg, set_config_key

setup_totalseg()
set_config_key("statistics_disclaimer_shown", True)

tasks = {
    "heartchambers_highres": [301],
    "total": [291, 292, 293, 294, 295, 298],
    "total_fast": [297],  # cropping model used internally by TS
    "trunk_cavities": [343],
}

for name, ids in tasks.items():
    for tid in ids:
        print(f"[build] Downloading {name} (task {tid})...")
        download_pretrained_weights(tid)

print("[build] All model weights downloaded.")
