import {
  video, overlay, statusEl, btnCam, btnCal,
  handLandmarker, lastTip, setLastTip, lastTime, setLastTime,
  VELOCITY_THRESH, wasInside
} from './state.js';
import { initAudio, play } from './audio.js';
import { initHands } from './hands.js';
import { loadOpenCVOnce } from './cv-loader.js';
import { findSquaresAndHomographyFromCurrentFrame, mapOverlayToSheet } from './calibration.js';
import { padsForScreen, tipToOverlayPx, overlayPxToSheet } from './pads.js';
import { renderOverlay } from './render.js';

// Overlay size sync
function resizeCanvas() {
  overlay.width  = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
video.addEventListener('loadedmetadata', () => {
  overlay.width  = video.videoWidth  || overlay.clientWidth  || 640;
  overlay.height = video.videoHeight || overlay.clientHeight || 480;
});
resizeCanvas();

// Camera
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();
}

// Main loop (unchanged logic)
async function loop(ts) {
  if (!video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(loop);
    return;
  }

  const result = handLandmarker?.detectForVideo?.(video, ts);

  let tipSheet = null;
  let tipPx    = null;

  if (result && result.landmarks && result.landmarks.length > 0) {
    const lm  = result.landmarks[0];
    const tip = lm[8];
    const p   = tipToOverlayPx(tip.x, tip.y);
    tipPx = p;
    tipSheet = mapOverlayToSheet(p.px, p.py) || overlayPxToSheet(p.px, p.py);
  }

  if (tipSheet) {
    const dt = (ts - lastTime) / 1000;
    let v = 0;
    if (lastTip && dt > 0) {
      const dx = tipSheet.x - lastTip.x;
      const dy = tipSheet.y - lastTip.y;
      v = Math.hypot(dx, dy) / dt;
    }

    if (v > VELOCITY_THRESH) {
      const pads = padsForScreen();
      for (const p of pads) {
        const d      = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
        const inside = d <= p.r;
        const prev   = wasInside.get(p.name) || false;
        if (inside && !prev) {
          const vol = Math.min(1.0, Math.max(0.15, v / 220));
          play(p.name, vol);
        }
        wasInside.set(p.name, inside);
      }
    } else {
      const pads = padsForScreen();
      for (const p of pads) {
        const d = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
        wasInside.set(p.name, d <= p.r);
      }
    }

    setLastTip(tipSheet);
  }

  setLastTime(ts);
  renderOverlay(tipPx);
  requestAnimationFrame(loop);
}

// Buttons
btnCam.onclick = async () => {
  await initCamera();
  await initAudio();
  await initHands();
  statusEl.textContent = "Camera on — show the printed sheet and tap a pad.";
  requestAnimationFrame(loop);
};

btnCal.onclick = async () => {
  statusEl.textContent = "Calibrating...";
  try {
    await loadOpenCVOnce();
    const ok = findSquaresAndHomographyFromCurrentFrame();
    statusEl.textContent = ok ? "Calibrated" : "Calibration failed";
  } catch (e) {
    console.error("Calibration error", e);
    statusEl.textContent = "Calibration error (see console)";
  }
};
