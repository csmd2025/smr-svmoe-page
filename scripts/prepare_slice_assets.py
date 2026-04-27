"""Prepare slice-section assets from re-trained model VTPs.

1. Side-view silhouette PNG (X vs Z scatter from full mesh) for ID & OOD.
2. 5 z-slabs per case showing GT/Pred/|Error| for both models, plus a
   diverging Δ|err| panel.
3. Slice metadata JSON for the front-end slider marker.

This version reads from the full-mesh inference VTPs (which contain all
3M points with vel_mag_gt/pred/error) and slices a thin slab around each
Z position, rather than reading pre-computed .vti slice grids.
"""
from pathlib import Path
import json
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

pv.OFF_SCREEN = True

# ParaView "Cool to Warm (Extended)" — must match COLOR_STOPS in js/viewer.js
COOL_WARM_EXT = LinearSegmentedColormap.from_list("cool_warm_ext", [
    (0.000, (0.085, 0.094, 0.490)),
    (0.150, (0.231, 0.298, 0.752)),
    (0.350, (0.553, 0.683, 0.883)),
    (0.500, (0.870, 0.870, 0.870)),
    (0.650, (0.945, 0.580, 0.480)),
    (0.850, (0.706, 0.016, 0.149)),
    (1.000, (0.404, 0.000, 0.121)),
])

ROOT = Path(__file__).resolve().parents[1]
SLICE_DIR = ROOT / "data" / "slices"
SIDE_DIR = ROOT / "data" / "side"
SLICE_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)

# New: re-trained models (E1_vanilla_redo, E3_svmoe_seed7) full inference output
FULL = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_relaunch_full_vtp"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_relaunch_full_vtp"),
}
CASES = {"id": "run_10", "ood": "run_4"}

FIELD_GT = "vel_mag_gt"
FIELD_PRED = "vel_mag_pred"
FIELD_ERR = "vel_mag_error"

PRED_CLIM = (0.0, 3.0)   # GT and prediction (velocity magnitude)
ERR_CLIM = (0.0, 0.7)    # absolute error
N_SLICES = 5             # 5 Z slabs per case


def render_planar(mesh, field, outpath, clim, cmap=COOL_WARM_EXT, point_size=4):
    """Render a point cloud as a 2D planar projection."""
    a = np.asarray(mesh.point_data[field])
    a_clean = np.where(np.isfinite(a), a, 0.0)
    a_clean = np.clip(a_clean, -50.0, 50.0).astype(np.float32)
    mesh = mesh.copy()
    mesh.point_data[field] = a_clean

    pl = pv.Plotter(off_screen=True, window_size=[640, 640])
    pl.background_color = "#0f172a"
    pl.add_mesh(
        mesh, scalars=field, clim=clim, cmap=cmap,
        show_scalar_bar=False, render_points_as_spheres=False,
        point_size=point_size,
    )
    pl.view_xy()
    pl.camera.zoom(1.3)
    pl.screenshot(str(outpath), transparent_background=False)
    pl.close()


def downsize_to_jpg(png_path, jpg_path):
    img = Image.open(png_path).convert("RGB")
    img.thumbnail((480, 480), Image.LANCZOS)
    img.save(jpg_path, "JPEG", quality=82, optimize=True)
    png_path.unlink(missing_ok=True)


def slice_slab(full_mesh, z_center, slab_thickness):
    """Extract points within a thin slab around z_center."""
    pts = np.asarray(full_mesh.points)
    mask = np.abs(pts[:, 2] - z_center) < (slab_thickness / 2)
    if mask.sum() == 0:
        return None
    sub = pv.PolyData(pts[mask].astype(np.float32))
    for k in [FIELD_GT, FIELD_PRED, FIELD_ERR]:
        sub.point_data[k] = np.asarray(full_mesh.point_data[k])[mask]
    return sub


# --------------------------------------------------------------
# 1. Side-view silhouette + slice metadata
# --------------------------------------------------------------
metadata = {}
for case, run_dir in CASES.items():
    src = FULL["e1"] / f"{run_dir}_volume_full.vtp"
    mesh = pv.read(str(src))
    pts = np.asarray(mesh.points)
    n = len(pts)
    pick = np.random.default_rng(0).choice(n, size=min(40_000, n), replace=False)
    x = pts[pick, 0]
    z = pts[pick, 2]

    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())

    # Choose 5 evenly-spaced Z positions covering the data range
    z_positions = np.linspace(zmin + (zmax - zmin) * 0.1, zmin + (zmax - zmin) * 0.9, N_SLICES).tolist()

    # Side view: Z horizontal, X vertical
    fig, ax = plt.subplots(figsize=(4.5, 1.8), dpi=140)
    ax.scatter(z, x, s=0.4, c="#94a3b8", alpha=0.35, linewidths=0)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.tick_params(colors="#64748b", labelsize=8, length=2)
    ax.set_xlabel("Z (slice axis)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("X", color="#94a3b8", fontsize=9)
    plt.tight_layout(pad=0.4)
    out_path = SIDE_DIR / f"side_{case}.png"
    fig.savefig(out_path, facecolor="#0f172a", dpi=140)
    plt.close(fig)
    img = Image.open(out_path).convert("RGB")
    img.thumbnail((520, 260), Image.LANCZOS)
    img.save(out_path, "PNG", optimize=True)

    metadata[case] = {
        "x": [xmin, xmax],
        "z": [zmin, zmax],
        "slices": z_positions,
    }
    print(f"[side] {case} ({run_dir}): z=[{zmin:.2f},{zmax:.2f}], slices={[f'{v:.2f}' for v in z_positions]}")

(ROOT / "data" / "slice_meta.json").write_text(json.dumps(metadata, indent=2))

# --------------------------------------------------------------
# 2. Render slices from full VTP (slab around each Z position)
# --------------------------------------------------------------
for case, run_dir in CASES.items():
    e1_full = pv.read(str(FULL["e1"] / f"{run_dir}_volume_full.vtp"))
    e3_full = pv.read(str(FULL["e3"] / f"{run_dir}_volume_full.vtp"))
    z_positions = metadata[case]["slices"]
    z_extent = metadata[case]["z"][1] - metadata[case]["z"][0]
    slab_thickness = z_extent / N_SLICES * 0.6  # 60% of inter-slice spacing

    # Diff range — capped at 1.0 to keep meaningful contrast
    diff_vals = []
    for k in range(N_SLICES):
        zc = z_positions[k]
        s_e1 = slice_slab(e1_full, zc, slab_thickness)
        s_e3 = slice_slab(e3_full, zc, slab_thickness)
        if s_e1 is None or s_e3 is None:
            continue
        # Pair points by spatial nearest-neighbor (assume same point set)
        a1 = np.abs(np.asarray(s_e1.point_data[FIELD_ERR]))
        a3 = np.abs(np.asarray(s_e3.point_data[FIELD_ERR]))
        # truncate to common length (in case slab masks differ slightly)
        n = min(len(a1), len(a3))
        d = a1[:n] - a3[:n]
        d = d[np.isfinite(d)]
        if d.size:
            diff_vals.append(d)
    diff_abs = min(1.0, float(np.percentile(np.abs(np.concatenate(diff_vals)), 95))) if diff_vals else 1.0
    diff_clim = (-diff_abs, diff_abs)

    for k, zc in enumerate(z_positions):
        s_e1 = slice_slab(e1_full, zc, slab_thickness)
        s_e3 = slice_slab(e3_full, zc, slab_thickness)
        if s_e1 is None or s_e3 is None:
            print(f"  [skip] {case} z{k} no points in slab")
            continue

        # GT (shared from E1)
        render_planar(s_e1, FIELD_GT, SLICE_DIR / f"{case}_gt_z{k}.png", PRED_CLIM)
        downsize_to_jpg(SLICE_DIR / f"{case}_gt_z{k}.png", SLICE_DIR / f"{case}_gt_z{k}.jpg")

        for model, slab in [("e1", s_e1), ("e3", s_e3)]:
            render_planar(slab, FIELD_PRED, SLICE_DIR / f"{case}_{model}_pred_z{k}.png", PRED_CLIM)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_pred_z{k}.png", SLICE_DIR / f"{case}_{model}_pred_z{k}.jpg")
            slab_abs = slab.copy()
            slab_abs.point_data[FIELD_ERR] = np.abs(np.asarray(slab.point_data[FIELD_ERR]))
            render_planar(slab_abs, FIELD_ERR, SLICE_DIR / f"{case}_{model}_err_z{k}.png", ERR_CLIM)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_err_z{k}.png", SLICE_DIR / f"{case}_{model}_err_z{k}.jpg")

        # Diff panel
        a1 = np.abs(np.asarray(s_e1.point_data[FIELD_ERR]))
        a3 = np.abs(np.asarray(s_e3.point_data[FIELD_ERR]))
        n = min(len(a1), len(a3))
        d = a1[:n] - a3[:n]
        diff_slab = pv.PolyData(np.asarray(s_e1.points)[:n].astype(np.float32))
        diff_slab.point_data["err_diff"] = d.astype(np.float32)
        render_planar(diff_slab, "err_diff", SLICE_DIR / f"{case}_diff_z{k}.png", diff_clim, cmap="RdBu_r")
        downsize_to_jpg(SLICE_DIR / f"{case}_diff_z{k}.png", SLICE_DIR / f"{case}_diff_z{k}.jpg")

    print(f"[slices] {case} ({run_dir}): pred=(0,3), err=(0,0.7), diff=±{diff_abs:.3f}")

print("Done.")
