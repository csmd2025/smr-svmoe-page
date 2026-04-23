// Z-slice viewer: slider + side-view position indicator.
// Images are either relative URLs (server mode) or data: URIs (standalone).

const sliceState = { case: 'id', z: 2 };

function sliceURL(name) {
  if (window.__SLICE_DATA__) return window.__SLICE_DATA__[name] || '';
  return `data/slices/${name}.jpg`;
}

function sideURL(name) {
  if (window.__SIDE_DATA__) return window.__SIDE_DATA__[name] || '';
  return `data/side/${name}.png`;
}

function updateSlices() {
  const { case: c, z } = sliceState;
  document.getElementById('slice-gt').src        = sliceURL(`${c}_gt_z${z}`);
  document.getElementById('slice-e1').src        = sliceURL(`${c}_e1_pred_z${z}`);
  document.getElementById('slice-e1-err-img').src = sliceURL(`${c}_e1_err_z${z}`);
  document.getElementById('slice-e3').src        = sliceURL(`${c}_e3_pred_z${z}`);
  document.getElementById('slice-e3-err-img').src = sliceURL(`${c}_e3_err_z${z}`);
  const diff = document.getElementById('slice-diff-img');
  if (diff) diff.src = sliceURL(`${c}_diff_z${z}`);
  updateSideView();
}

function updateSideView() {
  const { case: c, z } = sliceState;
  const img = document.getElementById('side-img');
  if (img) img.src = sideURL(`side_${c}`);

  const meta = (window.__SLICE_META__ || {})[c];
  const marker = document.getElementById('side-marker');
  const label = document.getElementById('side-label');
  if (!meta || !marker) return;
  const [zmin, zmax] = meta.z;
  const zval = meta.slices[z];
  // Horizontal axis of the side-view PNG is Z (left=zmin, right=zmax)
  const pct = ((zval - zmin) / (zmax - zmin)) * 100;
  marker.style.left = `calc(${pct}% - 1px)`;
  if (label) label.textContent = `z = ${zval.toFixed(2)}  (slice ${z + 1}/5)`;
}

window.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('sliceSlider');
  if (slider) {
    slider.addEventListener('input', (e) => {
      sliceState.z = parseInt(e.target.value, 10);
      updateSlices();
    });
  }
  const caseBtns = document.getElementById('sliceCaseBtns');
  if (caseBtns) {
    caseBtns.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        caseBtns.querySelectorAll('button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        sliceState.case = btn.dataset.case;
        updateSlices();
      });
    });
  }
  updateSlices();
});
