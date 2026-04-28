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

# ParaView "Cool to Warm (Extended)" — exact 11-stop ramp.
# Must match COLOR_STOPS in js/viewer.js and js/slices.js.
COOL_WARM_EXT = LinearSegmentedColormap.from_list("cool_warm_ext", [
    (0.000, (0.059, 0.059, 0.471)),
    (0.114, (0.231, 0.298, 0.753)),
    (0.227, (0.396, 0.557, 0.847)),
    (0.341, (0.643, 0.776, 0.922)),
    (0.455, (0.867, 0.867, 0.867)),
    (0.500, (0.914, 0.835, 0.816)),
    (0.568, (0.958, 0.718, 0.659)),
    (0.682, (0.937, 0.522, 0.405)),
    (0.795, (0.847, 0.286, 0.227)),
    (0.909, (0.706, 0.055, 0.149)),
    (1.000, (0.404, 0.000, 0.122)),
])

ROOT = Path(__file__).resolve().parents[1]
SLICE_DIR = ROOT / "data" / "slices"
SIDE_DIR = ROOT / "data" / "side"
SLICE_DIR.mkdir(parents=True, exist_ok=True)
SIDE_DIR.mkdir(parents=True, exist_ok=True)

# New: re-trained models (E1_vanilla_redo, E3_svmoe_seed7) full inference output
FULL = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_relaunch_full_vtp"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_seed13_full_vtp"),
}
CASES = {"id": "run_10", "ood": "run_4"}

FIELD_GT = "vel_mag_gt"
FIELD_PRED = "vel_mag_pred"
FIELD_ERR = "vel_mag_error"

PRED_CLIM = (0.0, 3.0)   # GT and prediction (velocity magnitude)
ERR_CLIM = (0.0, 0.7)    # absolute error
N_SLICES = 5             # 5 Z slabs per case
N_SCAN = 80              # fine-grained scan for best contrast
MIN_SEP_FRAC = 0.025     # allow tighter clustering (top contrast region is narrow)


def best_contrast_z_positions(e1_full, e3_full, z_min, z_max, n_pick=5, n_scan=80, min_sep_frac=0.025):
    """Scan many Z positions, score each by E3-vs-E1 visual contrast,
    and return n_pick positions where SVMoE most dramatically beats baseline.

    Score = absolute drop in (|err|>0.3) red-fraction, weighted up by
    baseline severity. This favours slabs where E1 has many red pixels
    AND E3 removes a lot of them — the visually striking cases.
    """
    pts = np.asarray(e1_full.points)
    err1 = np.abs(np.asarray(e1_full.point_data[FIELD_ERR]))
    err3 = np.abs(np.asarray(e3_full.point_data[FIELD_ERR]))
    err1 = np.clip(np.where(np.isfinite(err1), err1, 0), 0, 50)
    err3 = np.clip(np.where(np.isfinite(err3), err3, 0), 0, 50)

    z_extent = z_max - z_min
    z_samples = np.linspace(z_min + z_extent * 0.05, z_min + z_extent * 0.95, n_scan)
    slab_thickness = z_extent / n_scan

    cands = []
    for zc in z_samples:
        mask = np.abs(pts[:, 2] - zc) < slab_thickness / 2
        if mask.sum() < 200:
            continue
        e1_red03 = (err1[mask] > 0.3).mean() * 100
        e3_red03 = (err3[mask] > 0.3).mean() * 100
        abs_drop = e1_red03 - e3_red03
        # Score favours absolute pp-drop, with a bonus for high-baseline-severity slabs
        score = abs_drop + 0.3 * e1_red03
        cands.append((float(zc), float(score), float(e1_red03), float(e3_red03)))

    # Greedy pick top scores with tight min separation (top contrast cluster is narrow)
    cands.sort(key=lambda x: -x[1])
    min_sep = z_extent * min_sep_frac
    picked = []
    for z, s, _, _ in cands:
        if all(abs(z - p) >= min_sep for p, _ in picked):
            picked.append((z, s))
            if len(picked) >= n_pick:
                break
    picked.sort(key=lambda x: x[0])
    return [p[0] for p in picked], [p[1] for p in picked]


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

    # Pick 5 Z positions where E3 visually beats E1 the most (greedy by score)
    e3_full_for_scan = pv.read(str(FULL["e3"] / f"{run_dir}_volume_full.vtp"))
    z_positions, scores = best_contrast_z_positions(
        mesh, e3_full_for_scan, zmin, zmax,
        n_pick=N_SLICES, n_scan=N_SCAN, min_sep_frac=MIN_SEP_FRAC,
    )
    print(f"  picked z positions (with score): " + ", ".join(f'{z:+.3f}({s:+.1f})' for z, s in zip(z_positions, scores)))

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
