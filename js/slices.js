// Z-slice viewer: slider + side-view position indicator.
// Images are either relative URLs (server mode) or data: URIs (standalone).

const sliceState = { case: 'id', z: 2 };
let sliceMeta = null;  // populated from window.__SLICE_META__ or fetched

// ParaView "Cool to Warm (Extended)" — shared with viewer.js
const SLICE_COLOR_STOPS = [
  [0.000, 0.059, 0.059, 0.471],
  [0.114, 0.231, 0.298, 0.753],
  [0.227, 0.396, 0.557, 0.847],
  [0.341, 0.643, 0.776, 0.922],
  [0.455, 0.867, 0.867, 0.867],
  [0.500, 0.914, 0.835, 0.816],
  [0.568, 0.958, 0.718, 0.659],
  [0.682, 0.937, 0.522, 0.405],
  [0.795, 0.847, 0.286, 0.227],
  [0.909, 0.706, 0.055, 0.149],
  [1.000, 0.404, 0.000, 0.122],
];

const SLICE_BARS = [
  { id: 'cb-pred', range: [0, 3.0], label: 'Velocity magnitude (GT and prediction)' },
  { id: 'cb-err',  range: [0, 0.7], label: '|Error| (per-model)' },
];

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

  const meta = (sliceMeta || {})[c];
  const marker = document.getElementById('side-marker');
  const label = document.getElementById('side-label');
  if (!meta || !marker) return;
  const [zmin, zmax] = meta.z;
  const zval = meta.slices[z];
  const bbox = meta.axes_bbox || { x: [0, 1], y: [0, 1] };
  const [bx0, bx1] = bbox.x;
  const [by0, by1] = bbox.y;
  const t = (zval - zmin) / (zmax - zmin);
  const pct = (bx0 + t * (bx1 - bx0)) * 100;
  marker.style.left = `calc(${pct}% - 1px)`;
  marker.style.top = `${(by0 * 100).toFixed(2)}%`;
  marker.style.height = `${((by1 - by0) * 100).toFixed(2)}%`;
  if (label) label.textContent = `z = ${zval.toFixed(2)}  (slice ${z + 1}/${meta.slices.length})`;
}

function renderSliceColorbars() {
  const host = document.getElementById('slice-colorbars');
  if (!host) return;
  const gradient = SLICE_COLOR_STOPS
    .map(([t, r, g, b]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(t * 100).toFixed(1)}%`)
    .join(', ');
  host.innerHTML = SLICE_BARS.map(({ range, label }) => {
    const [lo, hi] = range;
    const nTicks = 6;
    const ticks = [];
    for (let i = 0; i <= nTicks; i++) {
      const t = i / nTicks;
      const v = lo + t * (hi - lo);
      ticks.push(`<span style="left:${(t * 100).toFixed(2)}%">${v.toFixed(2)}</span>`);
    }
    return `
      <div class="slice-cb">
        <div class="cb-bar" style="background:linear-gradient(90deg, ${gradient})"></div>
        <div class="cb-ticks">${ticks.join('')}</div>
        <div class="cb-label">${label}</div>
      </div>`;
  }).join('');
}

async function loadSliceMeta() {
  if (window.__SLICE_META__) {
    sliceMeta = window.__SLICE_META__;
    return;
  }
  try {
    const res = await fetch('data/slice_meta.json');
    if (res.ok) sliceMeta = await res.json();
  } catch (e) {
    console.warn('[slices] failed to fetch slice_meta.json', e);
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  const slider = document.getElementById('sliceSlider');
  if (slider) {
    slider.addEventListener('input', (e) => {
      sliceState.z = parseInt(e.target.value, 10);
      updateSlices();
    });
  }
  const caseBtns = document.getElementById('sliceCaseBtns');
  const sliceGrid = document.querySelector('.slice-grid');
  if (caseBtns) {
    caseBtns.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', async () => {
        if (btn.classList.contains('active')) return;
        caseBtns.querySelectorAll('button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        sliceState.case = btn.dataset.case;
        if (sliceGrid && window.withInference) {
          await window.withInference(sliceGrid, () => updateSlices());
        } else {
          updateSlices();
        }
      });
    });
  }
  renderSliceColorbars();
  await loadSliceMeta();
  updateSlices();
});
