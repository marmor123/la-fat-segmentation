"""Batch TotalSegmentator Precomputation Runner for Remaining Patients.

Runs TS v2 (heartchambers_highres + trunk_cavities) sequentially on CPU
for any patient in the cohort manifest that does not yet have cached masks,
then automatically refreshes the benchmark results.
"""

import os
import sys
import time
import subprocess
from totalsegmentator.python_api import totalsegmentator

DATA_DIR = r"C:\Users\marmo\Downloads\ctscans"
PATIENT_IDS = ["2996", "3448", "6451", "8359", "8462", "9209"]

def run_ts_for_patient(pid: str) -> None:
    raw_ct_path = os.path.join(DATA_DIR, f"{pid}.nii.gz")
    if not os.path.isfile(raw_ct_path):
        raw_ct_path = os.path.join(DATA_DIR, f"{pid}.nii")
    if not os.path.isfile(raw_ct_path):
        print(f"[-] Raw CT not found for {pid}. Skipping.")
        return

    out_dir = os.path.join(DATA_DIR, "masks", pid)
    os.makedirs(out_dir, exist_ok=True)

    la_mask_path = os.path.join(out_dir, "heart_atrium_left.nii.gz")
    peri_mask_path = os.path.join(out_dir, "pericardium.nii.gz")

    if os.path.isfile(la_mask_path) and os.path.isfile(peri_mask_path):
        print(f"[+] Patient {pid} already has required masks. Skipping.")
        return

    print(f"\n=======================================================")
    print(f"[*] Starting TS Inference for Patient {pid}")
    print(f"=======================================================")

    # 1. heartchambers_highres
    if not os.path.isfile(la_mask_path):
        print(f"[+] Running task: heartchambers_highres (CPU)...")
        t0 = time.time()
        totalsegmentator(raw_ct_path, out_dir, task="heartchambers_highres", device="cpu")
        print(f"[+] heartchambers_highres completed in {time.time() - t0:.1f}s")

    # 2. trunk_cavities
    if not os.path.isfile(peri_mask_path):
        print(f"[+] Running task: trunk_cavities (CPU)...")
        t0 = time.time()
        totalsegmentator(raw_ct_path, out_dir, task="trunk_cavities", device="cpu")
        print(f"[+] trunk_cavities completed in {time.time() - t0:.1f}s")

    print(f"[+] Completed all TS masks for Patient {pid}!")

    # Refresh benchmark script
    print(f"[+] Refreshing Cohort Benchmark...")
    subprocess.run([sys.executable, "scripts/benchmark_10_patients.py"])

def main():
    print(f"Starting batch TS precomputation for {len(PATIENT_IDS)} patients: {PATIENT_IDS}")
    for pid in PATIENT_IDS:
        run_ts_for_patient(pid)
    print("\n[+] Batch TS precomputation finished for all patients!")

if __name__ == "__main__":
    main()
