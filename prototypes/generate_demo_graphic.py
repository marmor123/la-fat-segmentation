"""
Generates summary figure of the Ticket 5 QA Viewer Prototype variants for documentation.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def generate_demo_figure():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0d14')
    
    # Titles & Styling
    titles = [
        "Variant A: Radiology PACS Layout\n(3-Plane Orthogonal + Curtain Wipe + Layers)",
        "Variant B: Cohort Scorecard & Focus\n(Triage Table + Landmarks + Biometrics)",
        "Variant C: 3D Colleague Presentation Studio\n(Interactive 3D Mesh + Slide Deck)"
    ]
    
    for i, ax in enumerate(axes):
        ax.set_facecolor('#121824')
        ax.set_title(titles[i], color='#f1f5f9', fontsize=13, fontweight='bold', pad=15)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color('#26354a')
            spine.set_linewidth(1.5)

    # Variant A Mockup Drawing
    ax_a = axes[0]
    # Sidebar
    ax_a.add_patch(patches.Rectangle((0.02, 0.05), 0.25, 0.9, color='#182234', ec='#26354a'))
    ax_a.text(0.04, 0.88, "LAYERS", color='#94a3b8', fontsize=8, fontweight='bold')
    ax_a.text(0.04, 0.80, "☑ CT Base", color='#fff', fontsize=8)
    ax_a.text(0.04, 0.72, "☑ Pericardium", color='#22c55e', fontsize=8, fontweight='bold')
    ax_a.text(0.04, 0.64, "☑ TS Anchors", color='#ef4444', fontsize=8)
    ax_a.text(0.04, 0.56, "☑ LA Fat Mask", color='#facc15', fontsize=8, fontweight='bold')
    ax_a.text(0.04, 0.48, "☑ PV Tail (0HU)", color='#a855f7', fontsize=8)
    # Main Viewport (Axial)
    ax_a.add_patch(patches.Rectangle((0.30, 0.15), 0.45, 0.8, color='#000000', ec='#3b82f6', lw=1.5))
    ax_a.add_patch(patches.Ellipse((0.52, 0.55), 0.35, 0.45, color='#22c55e', fill=False, lw=2))
    ax_a.add_patch(patches.Ellipse((0.52, 0.62), 0.18, 0.20, color='#ef4444', alpha=0.5))
    ax_a.add_patch(patches.Ellipse((0.52, 0.64), 0.22, 0.24, color='#facc15', alpha=0.7, lw=2))
    ax_a.text(0.32, 0.88, "AXIAL Z: 34\nLA Fat: 22.4 mL", color='#facc15', fontsize=8, family='monospace')
    # Curtain Line
    ax_a.plot([0.52, 0.52], [0.15, 0.95], color='#facc15', lw=2, linestyle='--')
    ax_a.text(0.53, 0.20, "⇹ Curtain Wipe", color='#facc15', fontsize=7, fontweight='bold')
    # Orthogonal subviews
    ax_a.add_patch(patches.Rectangle((0.78, 0.55), 0.20, 0.40, color='#000000', ec='#26354a'))
    ax_a.text(0.80, 0.88, "CORONAL", color='#94a3b8', fontsize=7)
    ax_a.add_patch(patches.Rectangle((0.78, 0.08), 0.20, 0.40, color='#000000', ec='#26354a'))
    ax_a.text(0.80, 0.41, "SAGITTAL", color='#94a3b8', fontsize=7)

    # Variant B Mockup Drawing
    ax_b = axes[1]
    # Top Scorecard Table
    ax_b.add_patch(patches.Rectangle((0.02, 0.70), 0.96, 0.25, color='#182234', ec='#26354a'))
    ax_b.text(0.05, 0.88, "COHORT SCORECARD (10 Patients)", color='#94a3b8', fontsize=8, fontweight='bold')
    ax_b.text(0.05, 0.78, "0674 | 22.4 mL | Std: 18.1 | +23.7% PV | 🟢 PASSED", color='#facc15', fontsize=8, family='monospace')
    ax_b.text(0.05, 0.72, "1512 | 38.6 mL | Std: 31.2 | +23.7% PV | 🟡 MED RATIO", color='#94a3b8', fontsize=8, family='monospace')
    # Left Biometrics Card
    ax_b.add_patch(patches.Rectangle((0.02, 0.05), 0.35, 0.60, color='#182234', ec='#26354a'))
    ax_b.text(0.05, 0.58, "ADAPTIVE LA FAT", color='#94a3b8', fontsize=8, fontweight='bold')
    ax_b.text(0.05, 0.46, "22.4 mL", color='#facc15', fontsize=18, fontweight='bold')
    ax_b.text(0.05, 0.36, "Gaussian: μ=-88, σ=18.5\nWindow: [-125, 0] HU\nPericardium: 🟢 TS Solid", color='#fff', fontsize=8, family='monospace')
    # Center Viewport + Landmarks
    ax_b.add_patch(patches.Rectangle((0.40, 0.28), 0.58, 0.37, color='#000000', ec='#3b82f6'))
    ax_b.add_patch(patches.Ellipse((0.69, 0.46), 0.25, 0.28, color='#facc15', alpha=0.7))
    # Filmstrip
    ax_b.text(0.40, 0.22, "LANDMARKS:", color='#94a3b8', fontsize=7, fontweight='bold')
    for idx, name in enumerate(["Apex", "Mid-LV", "Mid-LA", "Mitral", "Ao"]):
        ax_b.add_patch(patches.Rectangle((0.40 + idx*0.115, 0.05), 0.105, 0.14, color='#000000', ec='#26354a'))
        ax_b.text(0.41 + idx*0.115, 0.08, name, color='#cbd5e1', fontsize=6)

    # Variant C Mockup Drawing
    ax_c = axes[2]
    # 3D Canvas
    ax_c.add_patch(patches.Rectangle((0.02, 0.05), 0.58, 0.9, color='#000000', ec='#3b82f6'))
    ax_c.add_patch(patches.Ellipse((0.31, 0.52), 0.40, 0.55, color='#22c55e', alpha=0.2, lw=1.5, ec='#22c55e'))
    ax_c.add_patch(patches.Ellipse((0.31, 0.60), 0.22, 0.26, color='#ef4444', alpha=0.6))
    ax_c.add_patch(patches.Ellipse((0.31, 0.63), 0.26, 0.30, color='#facc15', alpha=0.8))
    ax_c.text(0.05, 0.88, "3D HEART ORBIT VIEW", color='#3b82f6', fontsize=8, fontweight='bold')
    ax_c.text(0.05, 0.82, "[Anterior] [Posterior] [Lateral] [4-Chamber]", color='#94a3b8', fontsize=7)
    # Presentation Cards
    ax_c.add_patch(patches.Rectangle((0.63, 0.62), 0.35, 0.33, color='#182234', ec='#facc15', lw=1.5))
    ax_c.text(0.65, 0.88, "1. Zero Septal Bleed", color='#fff', fontsize=8, fontweight='bold')
    ax_c.text(0.65, 0.70, "Multi-anchor distance\ntransforms eliminate\nventricular spillover.", color='#94a3b8', fontsize=7)
    ax_c.add_patch(patches.Rectangle((0.63, 0.26), 0.35, 0.33, color='#182234', ec='#a855f7', lw=1.5))
    ax_c.text(0.65, 0.52, "2. +23.7% PV Recovery", color='#fff', fontsize=8, fontweight='bold')
    ax_c.text(0.65, 0.34, "Adaptive 0 HU clamp\nrecovers boundary\npartial volume fat.", color='#94a3b8', fontsize=7)
    ax_c.add_patch(patches.Rectangle((0.63, 0.05), 0.35, 0.18, color='#3b82f6'))
    ax_c.text(0.66, 0.12, "📑 Export Report", color='#fff', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_dir = "docs/prototypes"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ticket_5_qa_viewer_demo.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated demo figure: {out_path}")

if __name__ == "__main__":
    generate_demo_figure()
