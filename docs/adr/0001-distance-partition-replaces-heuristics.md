# Partition by nearest surface distance replaces explicit anatomical heuristics

The original pipeline (v1–v3) used four separate heuristic steps to carve LA fat from epicardial fat: an SVM-based mitral plane cutoff to separate LA from LV, dilate-and-subtract exclusions for RA/aorta/PA to prevent fat bleed, a superior buffer below the LA dome to prevent great-vessel fat inclusion, and a pulmonary vein plug to close the posterior pericardial reflection. Each step was developed in response to specific failures on specific scans, creating a fragile chain of patches.

We replaced all four with a single distance-based partition: compute the 3D distance transform from each of six anchor surfaces (LA, LV, RA, RV, Aorta, Pulmonary Artery), then assign each epicardial fat voxel to whichever anchor it is nearest. LA fat = all fat voxels whose nearest anchor is the LA.

This was chosen over the heuristic pipeline because (a) it's anatomically principled — a radiologist mentally partitions epicardial fat by proximity to each chamber wall, (b) it naturally handles tilted hearts and anatomical variants that the explicit cutoffs couldn't, and (c) it's simpler to implement, debug, and iterate on. The cost is that it depends on all six TS masks being reasonably accurate — a bad mask skews the partition — but a degraded partition is easier to detect and flag than a silently-wrong heuristic.
