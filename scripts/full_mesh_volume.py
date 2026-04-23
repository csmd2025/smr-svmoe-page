"""Produce FULL-resolution VTPs (every mesh point) for the web viewer.

The raw run_N_volume_full.vtp files are ~280-413 MB because they store 15 redundant
arrays (three velocity components + their gt/pred/error + magnitudes + vectors).
The viewer only needs vel_mag_{gt,pred,error}, so we strip the rest and write a
zlib-compressed binary XML VTP.
"""
from pathlib import Path
import numpy as np
import pyvista as pv
import vtk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vtp"
OUT.mkdir(parents=True, exist_ok=True)

FULL_VTP_DIR = {
    "e1": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E1_full_vtp"),
    "e3": Path("/mnt/storage/sanghyeon-kim/HYU_preprocessing/outputs/E3_full_vtp"),
}
CASES = {"run4": "run_4", "run7": "run_7"}


def save_compact(points, gt, pr, err, out):
    cloud = pv.PolyData(points.astype(np.float32))
    cloud.point_data["vel_mag_gt"] = gt.astype(np.float32)
    cloud.point_data["vel_mag_pred"] = pr.astype(np.float32)
    cloud.point_data["vel_mag_error"] = err.astype(np.float32)

    # Write with appended+zlib for maximum compression.
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out))
    writer.SetInputData(cloud)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    writer.SetEncodeAppendedData(False)  # raw binary, not base64
    writer.Write()


total_before = total_after = 0
for case_key, run_dir in CASES.items():
    for model, root in FULL_VTP_DIR.items():
        src = root / f"{run_dir}_volume_full.vtp"
        if not src.exists():
            print(f"[skip] {src}")
            continue
        m = pv.read(str(src))
        pts = np.asarray(m.points)
        gt = np.asarray(m.point_data["vel_mag_gt"])
        pr = np.asarray(m.point_data["vel_mag_pred"])
        err = np.abs(np.asarray(m.point_data["vel_mag_error"]))
        before = src.stat().st_size

        out = OUT / f"{model}_{case_key}_volume.vtp"
        save_compact(pts, gt, pr, err, out)
        after = out.stat().st_size

        total_before += before
        total_after += after
        print(f"[{model} {case_key}] {m.n_points:,} pts: {before/1024/1024:.0f} MB -> {after/1024/1024:.1f} MB")

print(f"\nTotal: {total_before/1024/1024:.0f} MB -> {total_after/1024/1024:.1f} MB "
      f"({100*total_after/total_before:.1f}%)")
