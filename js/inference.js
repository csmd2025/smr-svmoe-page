// Fake "running inference" overlay with countdown + progress bar.
// Used to make case/field switches feel like a real model inference call.

function ensureInferenceOverlay(host) {
  let ov = host.querySelector(':scope > .inference-overlay');
  if (ov) return ov;
  ov = document.createElement('div');
  ov.className = 'inference-overlay';
  ov.innerHTML = `
    <div class="inference-spinner"></div>
    <div class="inference-label">Running inference<span class="dots"></span></div>
    <div class="inference-progress"><div class="inference-progress-bar"></div></div>
    <div class="inference-eta">3.0s</div>
  `;
  host.appendChild(ov);
  return ov;
}

function showInferenceOverlay(host, durationMs) {
  const ov = ensureInferenceOverlay(host);
  const bar = ov.querySelector('.inference-progress-bar');
  const eta = ov.querySelector('.inference-eta');
  bar.style.width = '0%';
  ov.classList.add('visible');

  return new Promise((resolve) => {
    const start = performance.now();
    const tick = () => {
      const elapsed = performance.now() - start;
      const t = Math.min(elapsed / durationMs, 1);
      bar.style.width = (t * 100).toFixed(1) + '%';
      const remaining = Math.max(durationMs - elapsed, 0) / 1000;
      eta.textContent = remaining > 0.05 ? `${remaining.toFixed(1)}s` : 'finalizing…';
      if (t < 1) requestAnimationFrame(tick);
      else {
        ov.classList.remove('visible');
        resolve();
      }
    };
    requestAnimationFrame(tick);
  });
}

// Run `fn` while the overlay is up; resolves only after BOTH the work and
// the minimum overlay duration are done. 2.7–3.3s by default for natural feel.
async function withInference(host, fn, opts = {}) {
  const dur = opts.duration ?? (2700 + Math.random() * 600);
  await Promise.all([showInferenceOverlay(host, dur), Promise.resolve().then(fn)]);
}

window.withInference = withInference;
