"""Prepare slice-section assets:

1. Side-view silhouette PNG (X vs Z scatter from full mesh) for ID & OOD.
2. Re-render the 5 z-slices with a diverging "Δ|err| = |err_e1| - |err_e3|" panel
   so SVMoE-wins regions (positive, red) and E1-wins (negative, blue) are visible.
3. Export Z positions + bbox JSON used by the front-end to place the slider
   marker on the silhouette.
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

SRC = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_baseline_ep275"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_vmoe_ep383"),
}
FULL = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_full_vtp"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_full_vtp"),
}
CASES = {"id": "run_4", "ood": "run_7"}

# --------------------------------------------------------------
# 1. Side-view silhouette PNGs (X vs Z scatter of full mesh)
# --------------------------------------------------------------
metadata = {}
for case, run_dir in CASES.items():
    src = FULL["e1"] / f"{run_dir}_volume_full.vtp"
    mesh = pv.read(str(src))
    pts = np.asarray(mesh.points)
    # Subsample for plotting speed
    n = len(pts)
    pick = np.random.default_rng(0).choice(n, size=min(40_000, n), replace=False)
    x = pts[pick, 0]
    z = pts[pick, 2]

    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())

    # Side view: Z horizontal (vehicle length), X vertical (height)
    fig, ax = plt.subplots(figsize=(4.5, 1.8), dpi=140)
    ax.scatter(z, x, s=0.4, c="#94a3b8", alpha=0.35, linewidths=0)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("auto")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#64748b", labelsize=8, length=2)
    ax.set_xlabel("Z (slice axis)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("X", color="#94a3b8", fontsize=9)
    plt.tight_layout(pad=0.4)
    out_path = SIDE_DIR / f"side_{case}.png"
    fig.savefig(out_path, facecolor="#0f172a", dpi=140)
    plt.close(fig)
    # Shrink
    img = Image.open(out_path).convert("RGB")
    img.thumbnail((520, 260), Image.LANCZOS)
    img.save(out_path, "PNG", optimize=True)

    # Collect Z positions of existing 5 slices
    z_positions = []
    for k in range(5):
        s = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        z_positions.append(float(s.bounds[4]))
    metadata[case] = {
        "x": [xmin, xmax],
        "z": [zmin, zmax],
        "slices": z_positions,
    }
    print(f"[side] {case}: z=[{zmin:.2f}, {zmax:.2f}], slices={[f'{v:.2f}' for v in z_positions]}")

(ROOT / "data" / "slice_meta.json").write_text(json.dumps(metadata, indent=2))

# --------------------------------------------------------------
# 2. Re-render the 5 z-slices with an additional Δ|err| panel.
# --------------------------------------------------------------
FIELD_GT = "vel_mag_gt"
FIELD_PRED = "vel_mag_pred"
FIELD_ERR = "vel_mag_error"

# Fixed slice colorbar ranges (per user request).
PRED_CLIM = (0.0, 3.0)   # GT and prediction (velocity magnitude)
ERR_CLIM = (0.0, 0.7)    # absolute error


def render_planar(mesh, field, outpath, clim, cmap=COOL_WARM_EXT):
    # Grid points outside the mesh get NaN OR extrapolated outliers (values
    # in the hundreds/thousands) that smear across cells with
    # interpolate_before_map. Replace NaNs and clip to a physical range so the
    # colormap actually reflects the model's predictions.
    a = np.asarray(mesh.point_data[field])
    a_clean = np.where(np.isfinite(a), a, 0.0)
    a_clean = np.clip(a_clean, -50.0, 50.0).astype(np.float32)
    mesh = mesh.copy()
    mesh.point_data[field] = a_clean

    pl = pv.Plotter(off_screen=True, window_size=[640, 640])
    pl.background_color = "#0f172a"
    pl.add_mesh(
        mesh, scalars=field, clim=clim, cmap=cmap,
        show_scalar_bar=False, interpolate_before_map=True,
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


for case, run_dir in CASES.items():
    # Symmetric range for the diff panel (|err_e1| - |err_e3|), use percentile
    # of absolute differences to avoid being skewed by outliers.
    diff_vals = []
    for k in range(5):
        me1 = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        me3 = pv.read(str(SRC["e3"] / run_dir / f"volume_slice_z_{k}.vti"))
        d = np.abs(np.asarray(me1.point_data[FIELD_ERR])) - np.abs(np.asarray(me3.point_data[FIELD_ERR]))
        d = d[np.isfinite(d)]
        if d.size:
            diff_vals.append(d)
    # Cap at 1.0 — most meaningful differences are <1, and extreme OOD outliers
    # would otherwise flatten the whole map.
    diff_abs = min(1.0, float(np.percentile(np.abs(np.concatenate(diff_vals)), 95))) if diff_vals else 1.0
    diff_clim = (-diff_abs, diff_abs)

    for k in range(5):
        # GT (shared — from E1)
        gt = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        render_planar(gt, FIELD_GT, SLICE_DIR / f"{case}_gt_z{k}.png", PRED_CLIM)
        downsize_to_jpg(SLICE_DIR / f"{case}_gt_z{k}.png", SLICE_DIR / f"{case}_gt_z{k}.jpg")

        # Per-model pred + err
        for model in ("e1", "e3"):
            m = pv.read(str(SRC[model] / run_dir / f"volume_slice_z_{k}.vti"))
            render_planar(m, FIELD_PRED, SLICE_DIR / f"{case}_{model}_pred_z{k}.png", PRED_CLIM)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_pred_z{k}.png", SLICE_DIR / f"{case}_{model}_pred_z{k}.jpg")
            render_planar(m, FIELD_ERR, SLICE_DIR / f"{case}_{model}_err_z{k}.png", ERR_CLIM)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_err_z{k}.png", SLICE_DIR / f"{case}_{model}_err_z{k}.jpg")

        # Diff panel: compute on E1 grid (same layout), store as extra array
        me1 = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        me3 = pv.read(str(SRC["e3"] / run_dir / f"volume_slice_z_{k}.vti"))
        d = np.abs(np.asarray(me1.point_data[FIELD_ERR])) - np.abs(np.asarray(me3.point_data[FIELD_ERR]))
        me1.point_data["err_diff"] = d.astype(np.float32)
        render_planar(me1, "err_diff", SLICE_DIR / f"{case}_diff_z{k}.png", diff_clim, cmap="RdBu_r")
        downsize_to_jpg(SLICE_DIR / f"{case}_diff_z{k}.png", SLICE_DIR / f"{case}_diff_z{k}.jpg")

    print(f"[slices] {case}: pred=(0,3), err=(0,0.7), diff=±{diff_abs:.3f}")

print("Done.")
