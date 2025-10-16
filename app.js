// Paper Drum — 6 Pads (Acoustic) — FIXED MAPPING + Y-FLIP FOR LABELS

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const btnCam = document.getElementById('btnCam');
const btnCal = document.getElementById('btnCal');
const cbMirror = document.getElementById('cbMirror'); // unchecked by default

// --- Coordinate space for the sheet overlay ---
const SHEET_W = 384, SHEET_H = 288;

// Base pad layout (defined for the PDF; origin effectively bottom-left)
const basePads = [
  { name: "Kick",    x:  64, y:  64, r: 34, sound: "sounds/kick.wav" },
  { name: "Snare",   x: 192, y:  64, r: 34, sound: "sounds/snare.wav" },
  { name: "HiHat C", x: 320, y:  64, r: 30, sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     x:  64, y: 180, r: 32, sound: "sounds/tom.wav" },
  { name: "Clap",    x: 192, y: 180, r: 32, sound: "sounds/clap.wav" },
  { name: "HiHat O", x: 320, y: 180, r: 30, sound: "sounds/hihat_open.wav" },
];
// OpenCV start
// --- OpenCV lazy loader (no <script> tag in HTML needed) ---
let cvReady = false;
let cvLoadPromise = null;
export async function loadOpenCVOnce() {
  if (cvReady) return;
  if (!cvLoadPromise) {
    cvLoadPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://docs.opencv.org/4.x/opencv.js';
      s.async = true;              // okay here; we'll wait on onRuntimeInitialized
      s.onload = () => {
        if (window.cv) {
          if (typeof cv.onRuntimeInitialized === 'function') {
            cv.onRuntimeInitialized = () => { cvReady = true; resolve(); };
          } else {
            // Some builds are already initialized
            cvReady = true; resolve();
          }
        } else {
          reject(new Error('OpenCV global not found after load'));
        }
      };
      s.onerror = (e) => reject(e);
      document.head.appendChild(s);
    });
  }
  return cvLoadPromise;
}
// Hidden canvas just for CV processing (not added to DOM)
const work = document.createElement('canvas');
work.width = 480;  // you can try 640x480 later if corners are small
work.height = 360;
const wctx = work.getContext('2d', { willReadFrequently: true });

//OpenCV Step 2 End
// v1.2 calibration (camera -> sheet homography)
let H = null; // 3x3 cv.Mat mapping from *work-canvas video coords* to *sheet coords*

function getFrameMatFromVideo(video) {
  // Draw the current video frame into the offscreen work canvas
  wctx.drawImage(video, 0, 0, work.width, work.height);
  // Create a Mat from the canvas (RGBA)
  const srcRgba = cv.imread(work);
  return srcRgba;
}


function findSquaresAndHomographyFromCurrentFrame(video) {
  if (!cvReady) return false;

  const srcRgba = getFrameMatFromVideo(video);        // RGBA
  const gray = new cv.Mat();
  cv.cvtColor(srcRgba, gray, cv.COLOR_RGBA2GRAY);

  const bin = new cv.Mat();
  cv.adaptiveThreshold(
    gray, bin, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
    31, 5
  );

  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kernel);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bin, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const cand = [];
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const peri = cv.arcLength(c, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(c, approx, 0.04 * peri, true);
    const area = cv.contourArea(approx);
    if (approx.rows === 4 && area > 150) {
      const r = cv.boundingRect(approx);
      const ratio = Math.max(r.width, r.height) / Math.max(1, Math.min(r.width, r.height));
      if (ratio < 1.4) {
        const M = cv.moments(approx, false);
        const cx = M.m10 / (M.m00 || 1), cy = M.m01 / (M.m00 || 1);
        const quad = [];
        for (let k = 0; k < 4; k++) {
          const x = approx.intPtr(k)[0], y = approx.intPtr(k)[1];
          quad.push({ x, y });
        }
        cand.push({ area, cx, cy, quad });
      }
    }
    approx.delete();
    c.delete();
  }

  // cleanup
  kernel.delete(); contours.delete(); hierarchy.delete();
  srcRgba.delete(); gray.delete(); bin.delete();

  if (cand.length < 4) {
    console.warn('Calibration: found fewer than 4 corner squares');
    return false;
  }

  // take top 4 by area and order TL, TR, BR, BL
  cand.sort((a,b)=> b.area - a.area);
  const four = cand.slice(0,4);
  four.sort((a,b)=> a.cy - b.cy);
  const top2 = four.slice(0,2).sort((a,b)=> a.cx - b.cx);
  const bot2 = four.slice(2,4).sort((a,b)=> a.cx - b.cx);
  const TL = top2[0], TR = top2[1], BR = bot2[1], BL = bot2[0];

  function rectFrom(sq){
    const xs = sq.quad.map(p=>p.x), ys = sq.quad.map(p=>p.y);
    return { minx: Math.min(...xs), maxx: Math.max(...xs), miny: Math.min(...ys), maxy: Math.max(...ys) };
  }
  const rTL = rectFrom(TL), rTR = rectFrom(TR), rBR = rectFrom(BR), rBL = rectFrom(BL);

  const src4 = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array([
    rTL.minx, rTL.miny,   // TL  (in work-canvas video coords)
    rTR.maxx, rTR.miny,   // TR
    rBR.maxx, rBR.maxy,   // BR
    rBL.minx, rBL.maxy    // BL
  ]));
  const dst4 = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array([
    0, 0,
    SHEET_W, 0,
    SHEET_W, SHEET_H,
    0, SHEET_H
  ]));

  const Hmat = cv.getPerspectiveTransform(src4, dst4);
  src4.delete(); dst4.delete();

  if (H) H.delete?.();
  H = Hmat;
  console.log('Calibration: homography set');
  return true;
}
//OpenCV Step 3
// OpenCV End


function syncOverlaySizeToVideo() {
  const v = document.getElementById('video');
  const c = document.getElementById('overlay');
  if (!v || !c) return;
  // Use the video’s *intrinsic* pixel size for crisp rendering
  c.width  = v.videoWidth  || v.clientWidth  || 640;
  c.height = v.videoHeight || v.clientHeight || 480;
}
video.addEventListener('loadedmetadata', syncOverlaySizeToVideo);
window.addEventListener('resize', syncOverlaySizeToVideo);


// Use this for all rendering & hit tests: invert Y once to match screen (top-left origin)
function padsForScreen() {
  return basePads.map(p => ({ ...p, y: (SHEET_H - p.y) }));
}

let audioCtx;
const samples = new Map();
let handLandmarker = null;
let lastTip = null;
let lastTime = 0;
const cooldown = new Map();
const COOLDOWN_MS = 120;
const VELOCITY_THRESH = 2.0;
let wasInside = new Map(); // padName -> boolean


function resizeCanvas() {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();




async function initAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  for (const p of basePads) {
    const ab = await fetch(p.sound).then(r => r.arrayBuffer());
    const buf = await audioCtx.decodeAudioData(ab);
    samples.set(p.name, buf);
  }
}

function play(name, gain=1.0) {
  if (!audioCtx) return;
  const now = performance.now();
  if (cooldown.get(name) && now - cooldown.get(name) < COOLDOWN_MS) return;
  const buf = samples.get(name);
  if (!buf) return;
  const src = audioCtx.createBufferSource();
  const g = audioCtx.createGain();
  g.gain.value = Math.max(0.1, Math.min(1.0, gain));
  src.buffer = buf;
  src.connect(g).connect(audioCtx.destination);
  src.start();
  cooldown.set(name, now);
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();
}


async function initHands() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    numHands: 1,
    runningMode: "VIDEO"
  });
}

btnCam.onclick = async () => {
  await initCamera();
  if (!audioCtx) await initAudio();
  if (!handLandmarker) await initHands();
  statusEl.textContent = "Camera on — show the printed sheet and tap a pad.";
  requestAnimationFrame(loop);
};

// Calibration verification message and check
btnCal.onclick = async () => {
  statusEl.textContent = "Calibrating...";
  try {
    await loadOpenCVOnce(); // lazy load; no-op after first time
    const ok = findSquaresAndHomographyFromCurrentFrame(video);
    statusEl.textContent = ok ? "Calibrated" : "Calibration failed";
    // visual breadcrumb on overlay
    const g = overlay.getContext('2d');
    g.save();
    g.strokeStyle = ok ? "#00ff66" : "#ff3355";
    g.lineWidth = 4;
    g.strokeRect(8,8, overlay.width-16, overlay.height-16);
    g.restore();
  } catch (e) {
    console.error("Calibration error", e);
    statusEl.textContent = "Calibration error (see console)";
  }
};


// --- Video → overlay mapping (accounts for object-fit: cover) ---
function getCoverMapping(overlayW, overlayH, videoW, videoH) {
  const scale = Math.max(overlayW / videoW, overlayH / videoH);
  const displayW = videoW * scale;
  const displayH = videoH * scale;
  const offsetX = (overlayW - displayW) / 2;
  const offsetY = (overlayH - displayH) / 2;
  return { displayW, displayH, offsetX, offsetY };
}

function tipToOverlayPx(tipNormX, tipNormY) {
  const overlayW = overlay.width;
  const overlayH = overlay.height;
  const videoW = video.videoWidth || overlayW;
  const videoH = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - tipNormX) : tipNormX; // mirror only if toggled
  const ny = tipNormY;

  const px = offsetX + nx * displayW;
  const py = offsetY + ny * displayH;
  return { px, py };
}

// Overlay pixels → sheet coords
function overlayPxToSheet(px, py) {
  const u = px / overlay.width;
  const v = py / overlay.height;
  return { x: u * SHEET_W, y: v * SHEET_H };
}

function renderOverlay(tipPx) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const pads = padsForScreen();   // use Y-flipped pads
  const sx = overlay.width / SHEET_W;
  const sy = overlay.height / SHEET_H;

  ctx.lineWidth = 2;
  for (const p of pads) {
    ctx.beginPath();
    ctx.arc(p.x * sx, p.y * sy, p.r * ((sx + sy) / 2), 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.stroke();
    ctx.font = "12px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillText(p.name, p.x * sx - 16, p.y * sy + 4);
  }

  if (tipPx) {
    ctx.beginPath();
    ctx.arc(tipPx.px, tipPx.py, 7, 0, Math.PI*2);
    ctx.fillStyle = "rgba(0,200,255,0.95)";
    ctx.fill();
  }
}

async function loop(ts) {
  if (!video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(loop);
    return;
  }
  const result = handLandmarker.detectForVideo(video, ts);
  let tipSheet = null;
  let tipPx = null;

  if (result && result.landmarks && result.landmarks.length > 0) {
    const lm = result.landmarks[0];
    const tip = lm[8]; // index fingertip
    const p = tipToOverlayPx(tip.x, tip.y);
    tipPx = p;
    tipSheet = overlayPxToSheet(p.px, p.py);
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
    const d = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
    const inside = d <= p.r;
    const prev = wasInside.get(p.name) || false;

    // Trigger only when we newly enter the circle AND moving fast enough
    if (inside && !prev) {
      // map speed to volume: tune the divisor (e.g., 220) for feel
      const vol = Math.min(1.0, Math.max(0.15, v / 220));
      play(p.name, vol);
    }
    wasInside.set(p.name, inside);
  }
} else {
  // Even when not fast, update hover state to prevent stuck "inside"
  const pads = padsForScreen();
  for (const p of pads) {
    const d = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
    wasInside.set(p.name, d <= p.r);
  }
}

    lastTip = tipSheet;
  }

  lastTime = ts;
  renderOverlay(tipPx);
  requestAnimationFrame(loop);
}
