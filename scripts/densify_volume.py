"""Build dense volume VTPs for the viewer.

Both run_4 (ID) and run_7 (OOD) now use the real full-mesh inference VTP
(2.65M / 3.84M points). We subsample to TARGET_POINTS with a fixed seed so that
E1 and E3 share the same spatial indices for fair side-by-side comparison.

Only vel_mag_{gt,pred,error} are kept and the output is compressed binary VTP.
"""
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vtp"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_POINTS = 200_000
SEED = 42

FULL_VTP_DIR = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_relaunch_full_vtp"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_seed13_full_vtp"),
}
# Fallback for runs that don't have the full-mesh inference yet.
GRID_ROOT = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_baseline_ep275"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_vmoe_ep383"),
}
CASES = {"run10": "run_10", "run4": "run_4"}
VEL_THRESHOLD = 0.02


def save_cloud(points, gt, pr, err, out):
    cloud = pv.PolyData(points.astype(np.float32))
    cloud.point_data["vel_mag_gt"] = gt.astype(np.float32)
    cloud.point_data["vel_mag_pred"] = pr.astype(np.float32)
    cloud.point_data["vel_mag_error"] = err.astype(np.float32)
    cloud.save(str(out))


def load_full_vtp(src):
    mesh = pv.read(str(src))
    pts = np.asarray(mesh.points)
    gt = np.asarray(mesh.point_data["vel_mag_gt"])
    pr = np.asarray(mesh.point_data["vel_mag_pred"])
    err = np.abs(np.asarray(mesh.point_data["vel_mag_error"]))
    return pts, gt, pr, err


def load_grid_as_cloud(src):
    grid = pv.read(str(src))
    gt = np.asarray(grid.point_data["velocity_gt"])
    pr = np.asarray(grid.point_data["velocity_pred"])
    pts = np.asarray(grid.points)
    vmag_gt = np.linalg.norm(gt, axis=-1)
    vmag_pr = np.linalg.norm(pr, axis=-1)
    err = np.abs(vmag_pr - vmag_gt)
    mask = vmag_gt > VEL_THRESHOLD
    valid = np.where(mask)[0]
    return pts[valid], vmag_gt[valid], vmag_pr[valid], err[valid]


# Shared subsample index per case so E1 and E3 pick the same points.
case_indices = {}

for case_key, run_dir in CASES.items():
    for model, root in FULL_VTP_DIR.items():
        full_src = root / f"{run_dir}_volume_full.vtp"
        if full_src.exists():
            pts, gt, pr, err = load_full_vtp(full_src)
            source = f"full-mesh {full_src.name}"
        else:
            grid_src = GRID_ROOT[model] / run_dir / "volume_grid.vti"
            pts, gt, pr, err = load_grid_as_cloud(grid_src)
            source = f"grid {grid_src.parent.name}/{grid_src.name}"

        if case_key not in case_indices:
            rng = np.random.default_rng(SEED + hash(case_key) % 1000)
            size = min(TARGET_POINTS, len(pts))
            case_indices[case_key] = rng.choice(len(pts), size=size, replace=False)
        idx = case_indices[case_key]
        if len(pts) != len(case_indices[case_key]) + 0:
            # Different runs may have different total points — regenerate the index
            # set per (case, point_count) pair. We cache by case alone because E1
            # and E3 should share the same mesh (same run). Guard here anyway.
            if len(pts) < idx.max() + 1:
                rng = np.random.default_rng(SEED + hash(case_key) % 1000)
                case_indices[case_key] = rng.choice(len(pts), size=min(TARGET_POINTS, len(pts)), replace=False)
                idx = case_indices[case_key]

        out = OUT / f"{model}_{case_key}_volume.vtp"
        save_cloud(pts[idx], gt[idx], pr[idx], err[idx], out)
        sz = out.stat().st_size / 1024 / 1024
        print(f"[{model} {case_key}] {source}  {len(pts):,} → {len(idx):,} pts  {sz:.2f} MB")

print("Done.")
