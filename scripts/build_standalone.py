"""Build a single-file standalone HTML with all assets inlined.

Output: dist/smr_project_page.html — double-click to open, no server needed.

The viewer.js and slices.js are mode-aware: when window.__VTP_DATA__ /
window.__SLICE_DATA__ / window.__SIDE_DATA__ are present they use those;
otherwise they fall back to fetch() against data/* URLs.
"""
import base64
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

html = (ROOT / "index.html").read_text()
css = (ROOT / "css" / "style.css").read_text()
viewer_js = (ROOT / "js" / "viewer.js").read_text()
slices_js = (ROOT / "js" / "slices.js").read_text()
framework_svg = (ROOT / "assets" / "framework.svg").read_text()
framework_svg = re.sub(r"^<\?xml[^>]+\?>\s*", "", framework_svg).strip()

# --------------------------------------------------------------
# Encode VTP files (full-mesh) as base64.
# --------------------------------------------------------------
vtp_dir = ROOT / "data" / "vtp"
vtp_data = {}
vtp_total = 0
for p in sorted(vtp_dir.glob("*.vtp")):
    buf = p.read_bytes()
    vtp_data[p.name] = base64.b64encode(buf).decode("ascii")
    vtp_total += len(buf)
    print(f"[vtp ] {p.name}: {len(buf)/1024/1024:.1f} MB")
print(f"       total {vtp_total/1024/1024:.1f} MB")

# --------------------------------------------------------------
# Slice JPGs as data URIs.
# --------------------------------------------------------------
slice_data = {}
for p in sorted((ROOT / "data" / "slices").glob("*.jpg")):
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    slice_data[p.stem] = f"data:image/jpeg;base64,{b64}"

# --------------------------------------------------------------
# Side-view PNGs.
# --------------------------------------------------------------
side_data = {}
for p in sorted((ROOT / "data" / "side").glob("*.png")):
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    side_data[p.stem] = f"data:image/png;base64,{b64}"

# --------------------------------------------------------------
# Slice metadata.
# --------------------------------------------------------------
slice_meta = json.loads((ROOT / "data" / "slice_meta.json").read_text())

# --------------------------------------------------------------
# VTK.js bundle.
# --------------------------------------------------------------
VTK_URL = "https://unpkg.com/vtk.js@32.0.0/vtk.js"
vtk_cache = ROOT / "scripts" / ".vtk_cache.js"
if not vtk_cache.exists():
    print(f"[vtk ] downloading {VTK_URL}")
    urllib.request.urlretrieve(VTK_URL, vtk_cache)
vtk_src = vtk_cache.read_text(encoding="utf-8", errors="replace")
print(f"[vtk ] {len(vtk_src)/1024/1024:.1f} MB")

# --------------------------------------------------------------
# Stitch HTML.
# --------------------------------------------------------------
html = html.replace(
    '<link rel="stylesheet" href="css/style.css">',
    f'<style>\n{css}\n</style>',
)
html = html.replace(
    '<img src="assets/framework.svg" alt="SVMoE framework">',
    framework_svg,
)

# Remove script tags that will be replaced with inlined versions.
html = re.sub(r'<script src="https://unpkg.com/vtk\.js[^"]*"></script>', '', html)
html = html.replace('<script src="js/viewer.js"></script>', '')
html = html.replace('<script src="js/slices.js"></script>', '')

data_block = (
    "<script>"
    f"window.__VTP_DATA__={json.dumps(vtp_data)};"
    f"window.__SLICE_DATA__={json.dumps(slice_data)};"
    f"window.__SIDE_DATA__={json.dumps(side_data)};"
    f"window.__SLICE_META__={json.dumps(slice_meta)};"
    "</script>"
)
scripts = (
    f"<script>\n{vtk_src}\n</script>"
    + data_block
    + f"<script>\n{viewer_js}\n</script>"
    + f"<script>\n{slices_js}\n</script>"
)
html = html.replace("</body>", scripts + "\n</body>")

out = DIST / "smr_project_page.html"
out.write_text(html, encoding="utf-8")
print(f"\nStandalone: {out} ({out.stat().st_size/1024/1024:.1f} MB)")
