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
from PIL import Image

pv.OFF_SCREEN = True

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


def global_range(case, field):
    lo, hi = np.inf, -np.inf
    for model in ("e1", "e3"):
        for k in range(5):
            f = SRC[model] / CASES[case] / f"volume_slice_z_{k}.vti"
            if not f.exists():
                continue
            m = pv.read(str(f))
            a = np.asarray(m.point_data[field])
            a = a[np.isfinite(a)]
            if a.size:
                lo = min(lo, float(a.min()))
                hi = max(hi, float(a.max()))
    return lo, hi


def render_planar(mesh, field, outpath, clim, cmap="turbo"):
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
    pred_clim = global_range(case, FIELD_PRED)
    err_clim = (0.0, global_range(case, FIELD_ERR)[1])
    # For diff: (|err_e1| - |err_e3|); symmetric around 0
    diff_abs = 0.0
    for k in range(5):
        me1 = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        me3 = pv.read(str(SRC["e3"] / run_dir / f"volume_slice_z_{k}.vti"))
        d = np.abs(np.asarray(me1.point_data[FIELD_ERR])) - np.abs(np.asarray(me3.point_data[FIELD_ERR]))
        d = d[np.isfinite(d)]
        if d.size:
            diff_abs = max(diff_abs, float(np.max(np.abs(d))))
    diff_clim = (-diff_abs * 0.8, diff_abs * 0.8)

    for k in range(5):
        # GT (shared — from E1)
        gt = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        render_planar(gt, FIELD_GT, SLICE_DIR / f"{case}_gt_z{k}.png", pred_clim)
        downsize_to_jpg(SLICE_DIR / f"{case}_gt_z{k}.png", SLICE_DIR / f"{case}_gt_z{k}.jpg")

        # Per-model pred + err
        for model in ("e1", "e3"):
            m = pv.read(str(SRC[model] / run_dir / f"volume_slice_z_{k}.vti"))
            render_planar(m, FIELD_PRED, SLICE_DIR / f"{case}_{model}_pred_z{k}.png", pred_clim)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_pred_z{k}.png", SLICE_DIR / f"{case}_{model}_pred_z{k}.jpg")
            render_planar(m, FIELD_ERR, SLICE_DIR / f"{case}_{model}_err_z{k}.png", err_clim)
            downsize_to_jpg(SLICE_DIR / f"{case}_{model}_err_z{k}.png", SLICE_DIR / f"{case}_{model}_err_z{k}.jpg")

        # Diff panel: compute on E1 grid (same layout), store as extra array
        me1 = pv.read(str(SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"))
        me3 = pv.read(str(SRC["e3"] / run_dir / f"volume_slice_z_{k}.vti"))
        d = np.abs(np.asarray(me1.point_data[FIELD_ERR])) - np.abs(np.asarray(me3.point_data[FIELD_ERR]))
        me1.point_data["err_diff"] = d.astype(np.float32)
        render_planar(me1, "err_diff", SLICE_DIR / f"{case}_diff_z{k}.png", diff_clim, cmap="RdBu_r")
        downsize_to_jpg(SLICE_DIR / f"{case}_diff_z{k}.png", SLICE_DIR / f"{case}_diff_z{k}.jpg")

    print(f"[slices] {case}: clim pred={pred_clim}, err top={err_clim[1]:.3f}, diff ±{diff_clim[1]:.3f}")

print("Done.")
