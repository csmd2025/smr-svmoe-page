// Z-slice viewer: slider + side-view position indicator.
// Images are either relative URLs (server mode) or data: URIs (standalone).

const sliceState = { case: 'id', z: 2 };
let sliceMeta = null;  // populated from window.__SLICE_META__ or fetched

// ParaView "Cool to Warm" (basic) — shared with viewer.js
const SLICE_COLOR_STOPS = [
  [0.000, 0.231, 0.298, 0.753],
  [0.250, 0.554, 0.683, 0.883],
  [0.500, 0.865, 0.865, 0.865],
  [0.750, 0.957, 0.586, 0.487],
  [1.000, 0.706, 0.016, 0.149],
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
  const pct = ((zval - zmin) / (zmax - zmin)) * 100;
  marker.style.left = `calc(${pct}% - 1px)`;
  if (label) label.textContent = `z = ${zval.toFixed(2)}  (slice ${z + 1}/5)`;
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
  renderSliceColorbars();
  await loadSliceMeta();
  updateSlices();
});
