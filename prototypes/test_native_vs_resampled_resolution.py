"""Experiment: Comparing Native Resolution (0.35x0.35x1.5mm) vs Resampled (1.5x1.5x1.5mm).

Properly accounts for NIfTI affine spatial orientations.
"""

from __future__ import annotations

import os
import sys
import time
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(__file__))
from prototype_partition_phantom import partition_solid_edt


def resample_mask_to_reference(source_img: nib.Nifti1Image, target_img: nib.Nifti1Image) -> np.ndarray:
    """Resample a source binary mask onto target image grid respecting world affine."""
    # Coordinate transformation from target voxel space to source voxel space:
    # x_source = inv(A_source) @ A_target @ x_target
    T = np.linalg.inv(source_img.affine) @ target_img.affine
    target_shape = target_img.shape
    
    # Process slice by slice to save memory
    resampled_mask = np.zeros(target_shape, dtype=np.uint8)
    source_data = np.asarray(source_img.dataobj, dtype=np.float32)
    
    for z in range(target_shape[2]):
        y_idx, x_idx = np.indices(target_shape[:2])
        z_idx = np.full_like(y_idx, z)
        homo = np.stack([y_idx.ravel(), x_idx.ravel(), z_idx.ravel(), np.ones(y_idx.size)])
        source_coords = (T @ homo)[:3]
        
        slice_resamp = map_coordinates(source_data, source_coords, order=0).reshape(target_shape[:2])
        resampled_mask[:, :, z] = (slice_resamp > 0).astype(np.uint8)
        
    return resampled_mask


def run_comparison(patient_id: str, raw_ct_path: str, ts_dir: str) -> None:
    print(f"\n{'='*80}\nResolution Comparison for Patient {patient_id}\n{'='*80}")
    
    # 1. Load Raw Native CT
    raw_img = nib.load(raw_ct_path)
    raw_ct = np.asarray(raw_img.dataobj, dtype=np.float32)
    raw_sp = tuple(float(s) for s in raw_img.header.get_zooms()[:3])
    vvol_raw = raw_sp[0] * raw_sp[1] * raw_sp[2] / 1000.0
    print(f"Native CT Grid: shape={raw_ct.shape}, spacing={raw_sp} mm, voxel_vol={vvol_raw:.6f} mL")

    # 2. Load 1.5mm Resampled CT and Masks
    resamp_ct_img = nib.load(os.path.join(ts_dir, "ct_resampled.nii.gz"))
    resamp_ct = np.asarray(resamp_ct_img.dataobj, dtype=np.float32)
    resamp_sp = tuple(float(s) for s in resamp_ct_img.header.get_zooms()[:3])
    vvol_15 = resamp_sp[0] * resamp_sp[1] * resamp_sp[2] / 1000.0
    print(f"Resampled CT Grid: shape={resamp_ct.shape}, spacing={resamp_sp} mm, voxel_vol={vvol_15:.6f} mL")

    # Load 1.5mm masks
    peri_img = nib.load(os.path.join(ts_dir, "pericardium.nii.gz"))
    peri_15 = np.asarray(peri_img.dataobj, dtype=np.uint8)

    name_map = {
        "LA": "heart_atrium_left.nii.gz",
        "LV": "heart_ventricle_left.nii.gz",
        "RA": "heart_atrium_right.nii.gz",
        "RV": "heart_ventricle_right.nii.gz",
        "Aorta": "aorta.nii.gz",
        "Pulmonary_Artery": "pulmonary_artery.nii.gz",
    }
    anchors_15 = {}
    for aname, fname in name_map.items():
        anchors_15[aname] = np.asarray(nib.load(os.path.join(ts_dir, fname)).dataobj, dtype=np.uint8)

    # Step A: 1.5mm Resampled Resolution
    t_resamp_start = time.perf_counter()
    chambers_15 = np.zeros_like(peri_15, dtype=bool)
    for m in anchors_15.values():
        chambers_15 |= (m > 0)
    fat_15_30 = ((peri_15 > 0) & (resamp_ct >= -190.0) & (resamp_ct <= -30.0) & ~chambers_15).astype(np.uint8)
    fat_15_0 = ((peri_15 > 0) & (resamp_ct >= -190.0) & (resamp_ct <= 0.0) & ~chambers_15).astype(np.uint8)

    la_fat_15_30, _, _ = partition_solid_edt(anchors_15, peri_15, fat_15_30, spacing=resamp_sp)
    la_fat_15_0, _, _ = partition_solid_edt(anchors_15, peri_15, fat_15_0, spacing=resamp_sp)
    t_resamp_total = time.perf_counter() - t_resamp_start

    # Step B: Native Resolution (Proper World Affine Resampling)
    t_native_start = time.perf_counter()
    print("Resampling masks to Native World Grid...")
    peri_native = resample_mask_to_reference(peri_img, raw_img)
    
    anchors_native = {}
    chambers_native = np.zeros_like(peri_native, dtype=bool)
    for aname, fname in name_map.items():
        img_obj = nib.load(os.path.join(ts_dir, fname))
        m_nat = resample_mask_to_reference(img_obj, raw_img)
        anchors_native[aname] = m_nat
        chambers_native |= (m_nat > 0)

    fat_nat_30 = ((peri_native > 0) & (raw_ct >= -190.0) & (raw_ct <= -30.0) & ~chambers_native).astype(np.uint8)
    fat_nat_0 = ((peri_native > 0) & (raw_ct >= -190.0) & (raw_ct <= 0.0) & ~chambers_native).astype(np.uint8)

    print("Running Partition on Native Grid...")
    la_fat_nat_30, _, _ = partition_solid_edt(anchors_native, peri_native, fat_nat_30, spacing=raw_sp)
    la_fat_nat_0, _, _ = partition_solid_edt(anchors_native, peri_native, fat_nat_0, spacing=raw_sp)
    t_native_total = time.perf_counter() - t_native_start

    # Comparison
    print(f"\n{'Metric':<35} | {'1.5mm Resampled Grid':<22} | {'Native 0.35mm Grid':<20}")
    print("-" * 82)
    print(f"{'Total EAT [-190, -30] HU':<35} | {np.count_nonzero(fat_15_30)*vvol_15:<19.2f} mL | {np.count_nonzero(fat_nat_30)*vvol_raw:<17.2f} mL")
    print(f"{'Total EAT [-190, 0] HU':<35} | {np.count_nonzero(fat_15_0)*vvol_15:<19.2f} mL | {np.count_nonzero(fat_nat_0)*vvol_raw:<17.2f} mL")
    print(f"{'LA Fat [-190, -30] HU':<35} | {np.count_nonzero(la_fat_15_30)*vvol_15:<19.2f} mL | {np.count_nonzero(la_fat_nat_30)*vvol_raw:<17.2f} mL")
    print(f"{'LA Fat [-190, 0] HU':<35} | {np.count_nonzero(la_fat_15_0)*vvol_15:<19.2f} mL | {np.count_nonzero(la_fat_nat_0)*vvol_raw:<17.2f} mL")
    print(f"{'Total Processing Time':<35} | {t_resamp_total:<19.2f} s  | {t_native_total:<17.2f} s")
    print("=" * 82)


if __name__ == "__main__":
    run_comparison(
        patient_id="0674",
        raw_ct_path=r"C:\Users\marmo\Downloads\ctscans\0674.nii.gz",
        ts_dir=r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\0674",
    )
    run_comparison(
        patient_id="3664",
        raw_ct_path=r"C:\Users\marmo\Downloads\ctscans\3664.nii.gz",
        ts_dir=r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\3664",
    )
