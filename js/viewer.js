// Interactive 3D viewer for E1 vs E3 volume point clouds.
// Uses vtkGenericRenderWindow so each viewer owns its own canvas, renderer,
// and interactor cleanly — fixes the "second viewer doesn't render" issue.

const G = vtk.Rendering.Misc.vtkGenericRenderWindow;
const vtkActor = vtk.Rendering.Core.vtkActor;
const vtkMapper = vtk.Rendering.Core.vtkMapper;
const vtkColorTransferFunction = vtk.Rendering.Core.vtkColorTransferFunction;
const { vtkXMLPolyDataReader } = vtk.IO.XML;

// Server mode (via HTTP) pulls full-resolution VTPs from data/vtp_full/.
// Standalone build overrides by injecting window.__VTP_DATA__ with the 200K
// subsample (keeps the single-file HTML load-able).
const DATA = {
  id:  { e1: 'data/vtp/e1_run10_volume.vtp', e3: 'data/vtp/e3_run10_volume.vtp' },
  ood: { e1: 'data/vtp/e1_run4_volume.vtp',  e3: 'data/vtp/e3_run4_volume.vtp' },
};

// ParaView "Cool to Warm (Extended)" — exact 11-stop ramp.
// Matches the reference figure colorbar: dark navy at low, near-white middle,
// dark crimson at high. Extended endpoints give cleaner visual contrast.
const COLOR_STOPS = [
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

// Fixed scalar ranges per field (so colorbar values are physically meaningful).
const FIELD_RANGES = {
  vel_mag_error: [0, 0.7],
  vel_mag_pred:  [0, 1.0],
  vel_mag_gt:    [0, 1.0],
};

const FIELD_LABELS = {
  vel_mag_error: '|Δv| velocity-magnitude error',
  vel_mag_pred:  'Predicted velocity magnitude',
  vel_mag_gt:    'Ground-truth velocity magnitude',
};

function showError(container, message) {
  container.innerHTML = `<div style="padding:40px;color:#94a3b8;text-align:center;font-size:13px;font-family:ui-monospace,Menlo,monospace">${message}</div>`;
}

function buildViewer(containerId) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error('[viewer] container not found:', containerId);
    return null;
  }
  container.classList.add('loading');

  let grw;
  try {
    grw = G.newInstance({ background: [0.06, 0.09, 0.15] });
    grw.setContainer(container);
    grw.resize();
  } catch (e) {
    console.error('[viewer] setup failed', containerId, e);
    showError(container, `WebGL setup failed: ${e.message}`);
    return null;
  }

  const renderer = grw.getRenderer();
  const renderWindow = grw.getRenderWindow();

  const actor = vtkActor.newInstance();
  const mapper = vtkMapper.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setPointSize(2);
  renderer.addActor(actor);

  const lut = vtkColorTransferFunction.newInstance();
  mapper.setLookupTable(lut);
  mapper.setUseLookupTableScalarRange(false);

  // Handle layout changes (tab switches, window resize)
  const ro = new ResizeObserver(() => grw.resize());
  ro.observe(container);

  return { container, grw, renderer, renderWindow, actor, mapper, lut };
}

function setScalars(ctx, polydata, fieldName) {
  const pd = polydata.getPointData();
  const arr = pd.getArrayByName(fieldName);
  if (!arr) return;
  pd.setActiveScalars(fieldName);
  const [lo, hi] = FIELD_RANGES[fieldName] || arr.getRange();
  ctx.lut.removeAllPoints();
  COLOR_STOPS.forEach(([t, r, g, b]) => ctx.lut.addRGBPoint(lo + t * (hi - lo), r, g, b));
  ctx.mapper.setScalarRange(lo, hi);
  ctx.mapper.setScalarModeToUsePointData();
  ctx.mapper.setColorModeToMapScalars();
}

// Inline base64 decoder (used only in standalone build — overwritten by build script)
function b64ToArrayBuffer(b64) {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes.buffer;
}

async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return await res.arrayBuffer();
}

async function loadVtp(ctx, source, fieldName) {
  if (!ctx) return;
  try {
    const buf = window.__VTP_DATA__
      ? b64ToArrayBuffer(source)
      : await fetchBuffer(source);
    const reader = vtkXMLPolyDataReader.newInstance();
    reader.parseAsArrayBuffer(buf);
    const polydata = reader.getOutputData(0);
    ctx.mapper.setInputData(polydata);
    setScalars(ctx, polydata, fieldName);
    ctx.renderer.resetCamera();
    ctx.renderWindow.render();
    ctx.container.classList.remove('loading');
    ctx.currentPolydata = polydata;
  } catch (e) {
    console.error('[viewer] load failed', e);
    showError(ctx.container, `Failed to load volume data: ${e.message}`);
  }
}

const state = { case: 'id', field: 'vel_mag_error' };
let viewerE1, viewerE3;

function resolveSource(caseKey, modelKey) {
  // Standalone build injects window.__VTP_DATA__ and viewer.js resolves to b64.
  if (window.__VTP_DATA__) {
    const basename = DATA[caseKey][modelKey].split('/').pop();
    return window.__VTP_DATA__[basename];
  }
  return DATA[caseKey][modelKey];
}

async function refresh() {
  if (!viewerE1 || !viewerE3) return;
  await Promise.all([
    loadVtp(viewerE1, resolveSource(state.case, 'e1'), state.field),
    loadVtp(viewerE3, resolveSource(state.case, 'e3'), state.field),
  ]);
  updateStats();
  renderColorbar();
}

function renderColorbar() {
  const cb = document.getElementById('colorbar');
  if (!cb) return;
  const [lo, hi] = FIELD_RANGES[state.field] || [0, 1];
  const label = FIELD_LABELS[state.field] || state.field;

  const gradient = COLOR_STOPS
    .map(([t, r, g, b]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(t * 100).toFixed(1)}%`)
    .join(', ');

  const nTicks = 8;
  const ticks = [];
  for (let i = 0; i <= nTicks; i++) {
    const t = i / nTicks;
    const v = lo + t * (hi - lo);
    ticks.push(`<span style="left:${(t * 100).toFixed(2)}%">${v.toFixed(2)}</span>`);
  }

  cb.innerHTML = `
    <div class="cb-bar" style="background:linear-gradient(90deg, ${gradient})"></div>
    <div class="cb-ticks">${ticks.join('')}</div>
    <div class="cb-label">${label}</div>
  `;
}

function updateStats() {
  const statsE1 = document.getElementById('stats-e1');
  const statsE3 = document.getElementById('stats-e3');
  const compute = (pd, name) => {
    const a = pd?.getPointData().getArrayByName(name);
    if (!a) return null;
    const [lo, hi] = a.getRange();
    const data = a.getData();
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += Math.abs(data[i]);
    return { min: lo, max: hi, mean: sum / data.length };
  };
  const fmt = (s) => s ? `min ${s.min.toFixed(3)} · mean ${s.mean.toFixed(3)} · max ${s.max.toFixed(3)}` : '—';
  if (statsE1) statsE1.textContent = fmt(compute(viewerE1?.currentPolydata, state.field));
  if (statsE3) statsE3.textContent = fmt(compute(viewerE3?.currentPolydata, state.field));
}

function wireButtons(groupId, key, onChange, onBefore) {
  const group = document.getElementById(groupId);
  if (!group) return;
  const grid = document.querySelector('.viewer-grid');
  group.querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (btn.classList.contains('active')) return;
      group.querySelectorAll('button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state[key] = btn.dataset[key];
      onBefore?.();
      if (grid && window.withInference) {
        await window.withInference(grid, () => refresh());
      } else {
        await refresh();
      }
      onChange?.();
    });
  });
}

// Linked-camera sync — rotate/zoom/pan in one viewer mirrors the other.
let isSyncing = false;
function syncCameraFromTo(srcCam, dst) {
  const c = dst.renderer.getActiveCamera();
  c.setPosition(...srcCam.getPosition());
  c.setFocalPoint(...srcCam.getFocalPoint());
  c.setViewUp(...srcCam.getViewUp());
  c.setParallelScale(srcCam.getParallelScale());
  c.setViewAngle(srcCam.getViewAngle());
  dst.renderer.resetCameraClippingRange();
  dst.renderWindow.render();
}

function linkCameras(a, b) {
  const camA = a.renderer.getActiveCamera();
  const camB = b.renderer.getActiveCamera();
  camA.onModified(() => {
    if (isSyncing) return;
    isSyncing = true;
    syncCameraFromTo(camA, b);
    isSyncing = false;
  });
  camB.onModified(() => {
    if (isSyncing) return;
    isSyncing = true;
    syncCameraFromTo(camB, a);
    isSyncing = false;
  });
}

// Rotate a 3-vector around a world axis (0=X, 1=Y, 2=Z) by `angle` radians.
// Right-hand rule: positive angle = counterclockwise when looking down +axis.
function rotateAroundWorldAxis(v, axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const [x, y, z] = v;
  if (axis === 0) return [x, c * y - s * z, s * y + c * z]; // around X
  if (axis === 1) return [c * x + s * z, y, -s * x + c * z]; // around Y
  return [c * x - s * y, s * x + c * y, z];                  // around Z
}

// Rotate camera position (relative to focal point) AND viewUp around a world
// axis through the focal point. This is a true world-axis orbit, unlike vtk's
// camera-frame azimuth/elevation/roll which are relative to viewUp/right/view.
// Snapshot/restore the full camera pose so we can return to the exact
// first-page-load view after the user has manually rotated/zoomed.
let initialCamState = null;
function captureCamState(viewer) {
  const c = viewer.renderer.getActiveCamera();
  return {
    position: Array.from(c.getPosition()),
    focalPoint: Array.from(c.getFocalPoint()),
    viewUp: Array.from(c.getViewUp()),
    parallelScale: c.getParallelScale(),
    viewAngle: c.getViewAngle(),
  };
}
function restoreCamState(viewer, s) {
  const c = viewer.renderer.getActiveCamera();
  c.setPosition(s.position[0], s.position[1], s.position[2]);
  c.setFocalPoint(s.focalPoint[0], s.focalPoint[1], s.focalPoint[2]);
  c.setViewUp(s.viewUp[0], s.viewUp[1], s.viewUp[2]);
  c.setParallelScale(s.parallelScale);
  c.setViewAngle(s.viewAngle);
  viewer.renderer.resetCameraClippingRange();
  viewer.renderWindow.render();
}

function rotateCameraAroundWorldAxis(camera, axis, angle) {
  const fp = camera.getFocalPoint();
  const pos = camera.getPosition();
  const up = camera.getViewUp();
  const rel = [pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]];
  const newRel = rotateAroundWorldAxis(rel, axis, angle);
  const newUp = rotateAroundWorldAxis(up, axis, angle);
  camera.setPosition(fp[0] + newRel[0], fp[1] + newRel[1], fp[2] + newRel[2]);
  camera.setViewUp(newUp[0], newUp[1], newUp[2]);
}

// Intro auto-rotation — two-phase sequence:
//   Phase 1: world +X axis by 90°  (tip the wedge into long-axis-vertical pose)
//   Phase 2: world +Z axis by 360° (full revolution around vertical axis)
// Rotates one camera; linked sync mirrors to the other. User interaction
// (pointerdown / wheel) cancels immediately. Calling while a previous
// rotation is in flight cancels the old one first.
let cancelCurrentAutoRotation = null;
function startAutoRotation(viewers, opts = {}) {
  const {
    tipDurationMs = 1200,
    spinDurationMs = 3000,
    tipDeg = 90,
    spinDeg = 360,
  } = opts;
  if (!viewers.every(Boolean)) return;
  if (cancelCurrentAutoRotation) cancelCurrentAutoRotation();

  const camera = viewers[0].renderer.getActiveCamera();
  let cancelled = false;
  const cancel = () => { cancelled = true; };
  cancelCurrentAutoRotation = cancel;
  viewers.forEach((v) => {
    v.container.addEventListener('pointerdown', cancel, { once: true });
    v.container.addEventListener('wheel', cancel, { once: true, passive: true });
  });

  const totalMs = tipDurationMs + spinDurationMs;
  const tipRadPerMs = (tipDeg * Math.PI / 180) / tipDurationMs;
  const spinRadPerMs = (spinDeg * Math.PI / 180) / spinDurationMs;
  const start = performance.now();
  let prev = start;
  function tick(now) {
    if (cancelled) return;
    const elapsed = now - start;
    if (elapsed >= totalMs) {
      cancelCurrentAutoRotation = null;
      return;
    }
    const dt = now - prev;
    prev = now;

    if (elapsed < tipDurationMs) {
      rotateCameraAroundWorldAxis(camera, 0, tipRadPerMs * dt);   // +X axis
    } else {
      rotateCameraAroundWorldAxis(camera, 2, spinRadPerMs * dt);  // +Z axis
    }

    viewers[0].renderer.resetCameraClippingRange();
    viewers[0].renderWindow.render();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

window.addEventListener('DOMContentLoaded', async () => {
  if (typeof vtk === 'undefined') {
    document.querySelectorAll('.viewer-canvas').forEach((c) =>
      showError(c, 'vtk.js did not load — check network or standalone build.')
    );
    return;
  }
  viewerE1 = buildViewer('viewer-e1');
  viewerE3 = buildViewer('viewer-e3');
  if (viewerE1 && viewerE3) linkCameras(viewerE1, viewerE3);
  // onBefore: restore the captured first-load camera pose so that loadVtp's
  // subsequent resetCamera() refits the new (ID/OOD) data along the original
  // view direction. onChange just animates — focalPoint/position are already
  // correctly set for the new data by then.
  const restoreInitialView = () => {
    if (initialCamState && viewerE1) restoreCamState(viewerE1, initialCamState);
  };
  const replayIntro = () => {
    if (viewerE1 && viewerE3) startAutoRotation([viewerE1, viewerE3]);
  };
  wireButtons('caseBtns', 'case', replayIntro, restoreInitialView);
  wireButtons('fieldBtns', 'field');
  await refresh();
  if (viewerE1 && viewerE3) {
    initialCamState = captureCamState(viewerE1);
    startAutoRotation([viewerE1, viewerE3]);
  }
});
