"""Prepare assets for the SMR project page.

1. Copy volume_3d.vtp for {E1, E3} × {run_4 (ID), run_7 (OOD)} into data/vtp/.
2. Render Z-slice PNGs (gt / pred / error) for both cases × 5 z positions × 2 models.
"""
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True  # required for headless rendering

ROOT = Path(__file__).resolve().parents[1]
DATA_VTP = ROOT / "data" / "vtp"
DATA_SLICES = ROOT / "data" / "slices"
DATA_VTP.mkdir(parents=True, exist_ok=True)
DATA_SLICES.mkdir(parents=True, exist_ok=True)

# Source inference outputs (one checkpoint per model, run_4 = ID, run_7 = OOD)
SRC = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_baseline_ep275"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_vmoe_ep383"),
}
CASES = {"id": "run_4", "ood": "run_7"}

# ------------------------------------------------------------
# 1. Volume VTPs (for 3D viewer)
# ------------------------------------------------------------
for case, run_dir in CASES.items():
    for model, root in SRC.items():
        src = root / run_dir / "volume_3d.vtp"
        dst = DATA_VTP / f"{model}_{run_dir.replace('run_', 'run')}_volume.vtp"
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        shutil.copy(src, dst)
        print(f"[vtp ] {dst.relative_to(ROOT)}")

# ------------------------------------------------------------
# 2. Z-slice PNGs — 5 positions × {gt, pred, err} × 2 models
# ------------------------------------------------------------
# We use the pre-generated volume_slice_z_{0..4}.vti and render three panels each:
#   - gt   (from either model, both share the same GT)
#   - pred (model-specific)
#   - err  (model-specific)
# GT is common so we write it once under {case}_gt_z{k}.png.

FIELD_PRED = "vel_mag_pred"
FIELD_GT = "vel_mag_gt"
FIELD_ERR = "vel_mag_error"

# Consistent colour ranges to make visual comparison honest.
# Compute a global range per case across both models and all z by first pass.
def global_range(case, field):
    lo, hi = np.inf, -np.inf
    for model, root in SRC.items():
        for k in range(5):
            f = root / CASES[case] / f"volume_slice_z_{k}.vti"
            if not f.exists():
                continue
            m = pv.read(f)
            a = np.asarray(m.point_data[field])
            lo = min(lo, float(a.min()))
            hi = max(hi, float(a.max()))
    return lo, hi


def render(mesh, field, outpath, clim):
    pl = pv.Plotter(off_screen=True, window_size=[640, 640])
    pl.background_color = "#0f172a"
    pl.add_mesh(
        mesh,
        scalars=field,
        clim=clim,
        cmap="turbo",
        show_scalar_bar=False,
        interpolate_before_map=True,
    )
    pl.view_xy()
    pl.camera.zoom(1.3)
    img = pl.screenshot(str(outpath), transparent_background=False, return_img=True)
    pl.close()
    return img


for case in CASES:
    run_dir = CASES[case]
    pred_clim = global_range(case, FIELD_PRED)
    err_clim = (0.0, global_range(case, FIELD_ERR)[1])  # errors clipped at 0 floor

    for k in range(5):
        # GT — shared (read from E1 since GT is identical)
        gt_src = SRC["e1"] / run_dir / f"volume_slice_z_{k}.vti"
        if not gt_src.exists():
            print(f"[skip] {gt_src}")
            continue
        gt_mesh = pv.read(gt_src)
        render(gt_mesh, FIELD_GT, DATA_SLICES / f"{case}_gt_z{k}.png", pred_clim)

        for model in ("e1", "e3"):
            src = SRC[model] / run_dir / f"volume_slice_z_{k}.vti"
            if not src.exists():
                print(f"[skip] {src}")
                continue
            mesh = pv.read(src)
            render(mesh, FIELD_PRED, DATA_SLICES / f"{case}_{model}_pred_z{k}.png", pred_clim)
            render(mesh, FIELD_ERR, DATA_SLICES / f"{case}_{model}_err_z{k}.png", err_clim)
        print(f"[png ] {case} z{k}")

print("Done.")
