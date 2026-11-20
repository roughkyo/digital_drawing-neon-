"""Analyze reference.png geometry, color palette, gradient energy, and landmark grid."""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "reference.png"
VIS_DIR = ROOT / "visualizations"
INT_DIR = ROOT / "intermediate_results"
VIS_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    img = Image.open(REF_PATH).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    print(f"Image size: {w}x{h}, aspect ratio: {w / h:.6f}")

    # Grayscale luminance (Rec. 709)
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Gradient magnitude & structure tensor orientation
    gx = ndimage.sobel(ndimage.gaussian_filter(gray, 1.5), axis=1)
    gy = ndimage.sobel(ndimage.gaussian_filter(gray, 1.5), axis=0)
    grad_mag = np.hypot(gx, gy)

    jxx = ndimage.gaussian_filter(gx * gx, 4.0)
    jxy = ndimage.gaussian_filter(gx * gy, 4.0)
    jyy = ndimage.gaussian_filter(gy * gy, 4.0)
    # Dominant tangent direction (perpendicular to gradient = along contours/forms)
    # Eigenvector of minimum eigenvalue of structure tensor
    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    theta_tangent = theta_grad + np.pi / 2.0
    coherence = np.hypot(jxx - jyy, 2.0 * jxy) / (jxx + jyy + 1e-6)

    # Save coordinate grid overlay for exact anatomical landmark mapping
    fig, ax = plt.subplots(figsize=(10, 15), dpi=130)
    ax.imshow(arr.astype(np.uint8))
    xticks = np.linspace(0, w, 17)
    yticks = np.linspace(0, h, 25)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x / w:.2f}" for x in xticks], fontsize=7, rotation=45)
    ax.set_yticklabels([f"{y / h:.2f}" for y in yticks], fontsize=7)
    ax.grid(color="cyan", linestyle="--", linewidth=0.6, alpha=0.65)
    ax.set_title(f"Reference Coordinate Grid ({w}x{h})", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_coordinate_grid.png")
    plt.close(fig)

    # Save zoomed face + chest badge crops with fine coordinate grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=130)
    # Face crop
    fx0, fx1, fy0, fy1 = int(0.28 * w), int(0.72 * w), int(0.15 * h), int(0.45 * h)
    axes[0].imshow(arr[fy0:fy1, fx0:fx1].astype(np.uint8), extent=[fx0 / w, fx1 / w, fy1 / h, fy0 / h])
    axes[0].set_title("Face & Beard Zoom (normalized coords)")
    axes[0].grid(color="yellow", linestyle=":", linewidth=0.6, alpha=0.7)

    # Chest badge + belt crop
    bx0, bx1, by0, by1 = int(0.28 * w), int(0.72 * w), int(0.47 * h), int(0.74 * h)
    axes[1].imshow(arr[by0:by1, bx0:bx1].astype(np.uint8), extent=[bx0 / w, bx1 / w, by1 / h, by0 / h])
    axes[1].set_title("Rank Badge & Belt Zoom (normalized coords)")
    axes[1].grid(color="yellow", linestyle=":", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_detail_zones.png")
    plt.close(fig)

    # Save gradient energy & flow visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), dpi=120)
    axes[0].imshow(np.log1p(grad_mag), cmap="magma")
    axes[0].set_title("Log Gradient Energy (Detail Density)")
    axes[0].axis("off")

    axes[1].imshow(coherence, cmap="viridis")
    step = 32
    yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    u = np.cos(theta_tangent[yy, xx]) * coherence[yy, xx]
    v = np.sin(theta_tangent[yy, xx]) * coherence[yy, xx]
    axes[1].quiver(xx, yy, u, v, color="white", angles="xy", scale=35, width=0.0025, headwidth=1, headlength=1)
    axes[1].set_title("Structure Tensor Coherence & Form Tangents")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_flow_and_energy.png")
    plt.close(fig)

    # Basic color & brightness summary per major band
    summary = {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 6),
        "mean_rgb": [round(float(x), 2) for x in arr.mean(axis=(0, 1))],
        "std_rgb": [round(float(x), 2) for x in arr.std(axis=(0, 1))],
        "bg_top_left_rgb": [round(float(x), 2) for x in arr[20:120, 20:120].mean(axis=(0, 1))],
        "bg_top_right_rgb": [round(float(x), 2) for x in arr[20:120, w - 120 : w - 20].mean(axis=(0, 1))],
        "bg_mid_left_rgb": [round(float(x), 2) for x in arr[int(0.3 * h) : int(0.38 * h), 20:120].mean(axis=(0, 1))],
        "face_cheek_rgb": [round(float(x), 2) for x in arr[int(0.24 * h) : int(0.28 * h), int(0.41 * w) : int(0.46 * w)].mean(axis=(0, 1))],
        "hat_crown_rgb": [round(float(x), 2) for x in arr[int(0.06 * h) : int(0.14 * h), int(0.42 * w) : int(0.58 * w)].mean(axis=(0, 1))],
        "robe_crimson_rgb": [round(float(x), 2) for x in arr[int(0.55 * h) : int(0.65 * h), int(0.15 * w) : int(0.25 * w)].mean(axis=(0, 1))],
        "mean_coherence": round(float(coherence.mean()), 4),
    }
    with open(INT_DIR / "01_reference_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
