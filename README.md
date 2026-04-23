# SMR Project Page — SVMoE AB-UPT

Static project page comparing Vanilla AB-UPT (E1) vs SVMoE AB-UPT (E3) on the HYU
internal-flow CFD dataset. Deployable to GitHub Pages.

## Layout
```
index.html            main page
css/style.css         layout / theme
js/viewer.js          VTK.js 3D viewer (E1 vs E3, ID vs OOD)
js/slices.js          Z-slice slider (PNG swap)
assets/framework.svg  SVMoE architecture diagram
scripts/prepare_data.py   Generate data/ assets from inference outputs
```

## Preparing assets

Data is kept out of git. Run the prep script once to populate `data/`:

```bash
/home/lab2_ksh/anaconda3/envs/ab-upt/bin/python scripts/prepare_data.py
```

This copies 4 × `volume_3d.vtp` files (~7 MB total) into `data/vtp/` and renders
30 × 640-pixel PNG slices into `data/slices/` (~10-15 MB total).

## Local preview
```bash
cd smr_project_page
python -m http.server 8000
# open http://localhost:8000
```

VTK.js loads `.vtp` via `fetch`, which requires an HTTP server — opening
`index.html` directly from the filesystem will fail with CORS/ACL errors.

## GitHub Pages
1. Push the folder to a repo (e.g. `smr-svmoe-page`).
2. Settings → Pages → Source: `main` / `/` (root).
3. Ensure `data/` is committed (the prep script output is ~20 MB, well under limits).
