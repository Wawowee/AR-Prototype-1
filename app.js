// =============================================================================
// Paper Drum — 6 Pads (Acoustic)
// v1.2: MediaPipe Hands + OpenCV homography (overlay <-> sheet)
// Organization-only refactor: NO functional changes
// =============================================================================

// -----------------------------------------------------------------------------
// Imports
// -----------------------------------------------------------------------------
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


window.addEventListener("load", () => console.log("app running(2)"));

// -----------------------------------------------------------------------------
// DOM Elements
// -----------------------------------------------------------------------------
const video    = document.getElementById('video');
const overlay  = document.getElementById('overlay');
const ctx      = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const btnCam   = document.getElementById('btnCam');
const btnCal   = document.getElementById('btnCal');
const cbMirror = document.getElementById('cbMirror'); // unchecked by default

// -----------------------------------------------------------------------------
// Constants: Sheet coordinate space & Pads
// -----------------------------------------------------------------------------
const SHEET_W = 620, SHEET_H = 400;
const PAD_SCALE = 1.53;

// Base pad layout (defined for the PDF; origin effectively bottom-left)
const basePads = [
  { name: "Kick",    x:  64, y:  72, r: 32, sound: "sounds/kick.wav" },
  { name: "Snare",   x: 192, y:  72, r: 32, sound: "sounds/snare.wav" },
  { name: "HiHat C", x: 320, y:  72, r: 32, sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     x:  64, y: 172, r: 32, sound: "sounds/tom.wav" },
  { name: "Clap",    x: 192, y: 172, r: 32, sound: "sounds/clap.wav" },
  { name: "HiHat O", x: 320, y: 172, r: 32, sound: "sounds/hihat_open.wav" },
];

// Convert basePads to screen/top-left sheet coords (single source of truth for draw + hit)
const TOP_ROW_DY = +6;   // + pushes top row down; - up
const BOT_ROW_DY = -10;   // + pushes bottom row down; - up
// scale factors from old (384×288) to new (620×400)
const SX = 620 / 384;
const SY = 400 / 288;
const SR = SX; // use width for circle radius scaling

function padsForScreen() {
  return basePads.map(p => {
    // scale pad center from old BL-origin sheet into the new sheet
    const xScaledBL = p.x * SX;
    const yScaledBL = p.y * SY;

    // flip Y to top-left origin in the new sheet space
    let yTL = SHEET_H - yScaledBL;

    // keep your per-row nudges (decide "top row" in BL-origin AFTER scaling)
    const isTopRow = yScaledBL > (SHEET_H / 2);
    yTL += isTopRow ? TOP_ROW_DY : BOT_ROW_DY;

    return {
      ...p,
      x: xScaledBL,
      y: yTL,
      r: Math.round(p.r * SR * PAD_SCALE)
    };
  });
}



// -----------------------------------------------------------------------------
// Runtime State (audio, hands, hit-test)
// -----------------------------------------------------------------------------
let audioCtx;
const samples = new Map();
let handLandmarker = null;

let lastTip  = null;
let lastTime = 0;

const cooldown       = new Map();
const COOLDOWN_MS    = 120;
const VELOCITY_THRESH= 2.0;
let wasInside        = new Map(); // padName -> boolean

// -----------------------------------------------------------------------------
// Overlay Sizing
// -----------------------------------------------------------------------------
function resizeCanvas() {
  overlay.width  = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function syncOverlaySizeToVideo() {
  const v = document.getElementById('video');
  const c = document.getElementById('overlay');
  if (!v || !c) return;
  // Use the video’s *intrinsic* pixel size for crisp rendering
  c.width  = v.videoWidth  || v.clientWidth  || 640;
  c.height = v.videoHeight || v.clientHeight || 480;
}
video.addEventListener('loadedmetadata', syncOverlaySizeToVideo);
window.addEventListener('resize',          syncOverlaySizeToVideo);

// -----------------------------------------------------------------------------
// Audio
// -----------------------------------------------------------------------------
async function initAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  for (const p of basePads) {
    const ab  = await fetch(p.sound).then(r => r.arrayBuffer());
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
  const g   = audioCtx.createGain();
  g.gain.value = Math.max(0.1, Math.min(1.0, gain));
  src.buffer = buf;
  src.connect(g).connect(audioCtx.destination);
  src.start();
  cooldown.set(name, now);
}

// -----------------------------------------------------------------------------
// Camera & Hands
// -----------------------------------------------------------------------------
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
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

async function ensureVideoReady() {
  if (video.videoWidth && video.videoHeight) return;
  await new Promise(res => {
    const onMeta = () => { video.removeEventListener('loadedmetadata', onMeta); res(); };
    video.addEventListener('loadedmetadata', onMeta);
  });
}

// -----------------------------------------------------------------------------
// Video <-> Overlay mapping helpers
// -----------------------------------------------------------------------------
function getCoverMapping(overlayW, overlayH, videoW, videoH) {
  const scale    = Math.max(overlayW / videoW, overlayH / videoH);
  const displayW = videoW * scale;
  const displayH = videoH * scale;
  const offsetX  = (overlayW - displayW) / 2;
  const offsetY  = (overlayH - displayH) / 2;
  return { displayW, displayH, offsetX, offsetY };
}

function tipToOverlayPx(tipNormX, tipNormY) {
  const overlayW = overlay.width;
  const overlayH = overlay.height;
  const videoW   = video.videoWidth || overlayW;
  const videoH   = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - tipNormX) : tipNormX; // mirror only if toggled
  const ny = tipNormY;

  return {
    px: offsetX + nx * displayW,
    py: offsetY + ny * displayH
  };
}

// Overlay pixels → sheet coords (v1.1 fallback)
function overlayPxToSheet(px, py) {
  const u = px / overlay.width;
  const v = py / overlay.height;
  return { x: u * SHEET_W, y: v * SHEET_H };
}

// Video pixel → overlay pixel
function videoPtToOverlayPx({x, y}) {
  const overlayW = overlay.width, overlayH = overlay.height;
  const videoW   = video.videoWidth || overlayW;
  const videoH   = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - x / videoW) : (x / videoW);
  const ny = y / videoH;

  return { px: offsetX + nx * displayW, py: offsetY + ny * displayH };
}

// -----------------------------------------------------------------------------
// OpenCV: Loader (lazy) + Work Canvas
// -----------------------------------------------------------------------------
let cvReady = false;
let cvLoadPromise = null;

 async function loadOpenCVOnce() {
  if (cvReady) return;
  if (cvLoadPromise) return cvLoadPromise;

  const sources = [
    'https://docs.opencv.org/4.x/opencv.js',
    // fallback (versioned mirror)
    'https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/platforms/js/opencv.js'
  ];

  cvLoadPromise = new Promise((resolve, reject) => {
    let idx = 0;

    const tryNext = () => {
      if (idx >= sources.length) {
        reject(new Error('Failed to load OpenCV.js from all sources'));
        return;
      }
      const url = sources[idx++];
      const s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onload = () => {
        const waitForAPI = () => {
          const ok = (typeof window.cv === 'object'
                   && typeof cv.Mat === 'function'
                   && typeof cv.getPerspectiveTransform === 'function');
          if (ok) { cvReady = true; resolve(); }
          else { setTimeout(waitForAPI, 25); }
        };
        if (window.cv && typeof cv.onRuntimeInitialized === 'function') {
          cv.onRuntimeInitialized = () => { cvReady = true; resolve(); };
          waitForAPI();
        } else {
          waitForAPI();
        }
      };
      s.onerror = () => {
        console.warn('OpenCV load failed from', url, '— trying fallback…');
        tryNext();
      };
      document.head.appendChild(s);
    };

    tryNext();
  });

  return cvLoadPromise;
}


// -----------------------------------------------------------------------------
// Calibration: Homography overlay<->sheet + helpers
// -----------------------------------------------------------------------------
let H    = null; // overlay -> sheet (cv.Mat 3x3)
let Hinv = null; // sheet   -> overlay (cv.Mat 3x3)
let lastOverlayCorners = null; // [{px,py} x 4] TL,TR,BR,BL

// Order arbitrary 4 points as TL,TR,BR,BL by sums/diffs
function orderCornersTLTRBRBL(pts) {
  const sum  = pts.map(p => p.x + p.y);
  const diff = pts.map(p => p.x - p.y);
  const TL = pts[sum.indexOf(Math.min(...sum))];
  const BR = pts[sum.indexOf(Math.max(...sum))];
  const TR = pts[diff.indexOf(Math.max(...diff))];
  const BL = pts[diff.indexOf(Math.min(...diff))];
  return [TL, TR, BR, BL];
}

// TL,TR,BR,BL overlay points -> homography (overlay -> sheet)
function computeHomographyOverlay(srcOverlayPts) {
  const dst = [
    {x:0, y:0}, {x:SHEET_W, y:0}, {x:SHEET_W, y:SHEET_H}, {x:0, y:SHEET_H}
  ];
  const srcMat = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    srcOverlayPts[0].px, srcOverlayPts[0].py,
    srcOverlayPts[1].px, srcOverlayPts[1].py,
    srcOverlayPts[2].px, srcOverlayPts[2].py,
    srcOverlayPts[3].px, srcOverlayPts[3].py,
  ]));
  const dstMat = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    dst[0].x, dst[0].y, dst[1].x, dst[1].y, dst[2].x, dst[2].y, dst[3].x, dst[3].y
  ]));
  const Hmat = cv.getPerspectiveTransform(srcMat, dstMat);
  srcMat.delete(); dstMat.delete();
  return Hmat;
}

// Project a single overlay px -> sheet coords using H
function mapOverlayToSheet(x, y) {
  if (!H || !cvReady) return null;
  const src = cv.matFromArray(1, 1, cv.CV_32FC2, new Float32Array([x, y]));
  const dst = new cv.Mat();
  cv.perspectiveTransform(src, dst, H);
  const out = { x: dst.data32F[0], y: dst.data32F[1] };
  src.delete(); dst.delete();
  return out;
}

// Reprojection RMS error: overlay -> sheet -> overlay should land back on the 4 corners
function rmsReprojError(overlayPts /* [{px,py} * 4] */) {
  if (!H || !Hinv || !cvReady) return Infinity;

  // overlay -> sheet
  const src = cv.matFromArray(
    4, 1, cv.CV_32FC2,
    new Float32Array(overlayPts.flatMap(p => [p.px, p.py]))
  );
  const mid = new cv.Mat();
  cv.perspectiveTransform(src, mid, H);

  // sheet -> overlay
  const back = new cv.Mat();
  cv.perspectiveTransform(mid, back, Hinv);

  let s = 0;
  for (let i = 0; i < 4; i++) {
    const dx = overlayPts[i].px - back.data32F[2*i + 0];
    const dy = overlayPts[i].py - back.data32F[2*i + 1];
    s += dx*dx + dy*dy;
  }
  src.delete(); mid.delete(); back.delete();
  return Math.sqrt(s / 4);
}

// -----------------------------------------------------------------------------
// OpenCV: Frame & Fiducial Detection
// -----------------------------------------------------------------------------
function frameToMat(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const c  = document.createElement('canvas');
  c.width  = vw; c.height = vh;
  const g  = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(video, 0, 0, vw, vh);
  return cv.imread(c); // CV_8UC4
}

// RotatedRect → 4 vertices, compatible across builds
function rotatedRectVertices(rr) {
  // Try RotatedRect.points
  if (cv.RotatedRect && typeof cv.RotatedRect.points === 'function') {
    const pts = cv.RotatedRect.points(rr);
    if (Array.isArray(pts) && pts.length === 4 && 'x' in pts[0]) return pts;
    if (pts && pts.data32F && pts.data32F.length >= 8) {
      const a = pts.data32F;
      const out = [
        { x: a[0], y: a[1] },
        { x: a[2], y: a[3] },
        { x: a[4], y: a[5] },
        { x: a[6], y: a[7] },
      ];
      pts.delete?.();
      return out;
    }
  }
  // Fallback: single-arg boxPoints(rr)
  if (typeof cv.boxPoints === 'function') {
    const m = cv.boxPoints(rr);
    if (m && m.data32F && m.data32F.length >= 8) {
      const a = m.data32F;
      const out = [
        { x: a[0], y: a[1] },
        { x: a[2], y: a[3] },
        { x: a[4], y: a[5] },
        { x: a[6], y: a[7] },
      ];
      m.delete?.();
      return out;
    }
  }
  throw new Error('Could not extract rotated-rect vertices from this OpenCV.js build');
}

// Contours → candidate squares (rotated boxes)
function detectSquareBoxesFullRes(src /* CV_8UC4 */) {
  const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const blur = new cv.Mat(); cv.GaussianBlur(gray, blur, new cv.Size(5,5), 0);
  const bin  = new cv.Mat(); cv.threshold(blur, bin, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);

  const contours = new cv.MatVector(), hierarchy = new cv.Mat();
  cv.findContours(bin, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const vw = src.cols, vh = src.rows, imgArea = vw * vh;
  const cands = [];

  for (let i=0;i<contours.size();i++) {
    const cnt  = contours.get(i);
    const peri = cv.arcLength(cnt, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, 0.04 * peri, true);
    if (approx.rows === 4 && cv.isContourConvex(approx)) {
      const area = cv.contourArea(approx);
      const af   = area / imgArea;
      if (af > 0.0005 && af < 0.3) {
        const r  = cv.boundingRect(approx);
        const ar = r.width / Math.max(1, r.height);
        if (ar > 0.45 && ar < 1.7) {
          const rr  = cv.minAreaRect(cnt);
          const box = rotatedRectVertices(rr);
          let cx = 0, cy = 0;
          for (const v of box) { cx += v.x; cy += v.y; }
          cx /= 4; cy /= 4;
          cands.push({ area, cx, cy, box });
        }
      }
    }
    approx.delete(); cnt.delete();
  }

  contours.delete(); hierarchy.delete(); gray.delete(); blur.delete(); bin.delete();
  return cands; // [{ area, cx, cy, box:[{x,y}*4] }, ...]
}

// Pick 4 most spread-out candidates (by centers)
function chooseFourMostSpread(cands) {
  if (cands.length < 4) return null;
  const top = [...cands].sort((a,b)=>b.area-a.area).slice(0,8);
  let best = null, bestScore = -1;
  for (let i=0;i<top.length;i++)
    for (let j=i+1;j<top.length;j++)
      for (let k=j+1;k<top.length;k++)
        for (let l=k+1;l<top.length;l++) {
          const set = [top[i], top[j], top[k], top[l]];
          let s = 0;
          for (let a=0;a<4;a++) for (let b=a+1;b<4;b++) {
            const dx=set[a].cx-set[b].cx, dy=set[a].cy-set[b].cy; s += dx*dx + dy*dy;
          }
          if (s > bestScore) { bestScore = s; best = set; }
        }
  return best;
}

// Order 4 points as TL,TR,BR,BL (sum/diff heuristic)
function orderTLTRBRBLBySumDiff(pts /* [{x,y}*4] */) {
  const sum  = pts.map(p => p.x + p.y);
  const diff = pts.map(p => p.x - p.y);
  const TL = pts[sum.indexOf(Math.min(...sum))];
  const BR = pts[sum.indexOf(Math.max(...sum))];
  const TR = pts[diff.indexOf(Math.max(...diff))];
  const BL = pts[diff.indexOf(Math.min(...diff))];
  return [TL, TR, BR, BL];
}

// Main: find squares & compute H/Hinv (overlay <-> sheet)
function findSquaresAndHomographyFromCurrentFrame(video) {
  if (!cvReady) return false;

  const src   = frameToMat(video);
  const cands = detectSquareBoxesFullRes(src);
  src.delete();

  if (!cands || cands.length < 4) return false;

  const four = chooseFourMostSpread(cands);
  if (!four) return false;

  // Sheet center from the four centers
  const cxMean = (four[0].cx + four[1].cx + four[2].cx + four[3].cx) / 4;
  const cyMean = (four[0].cy + four[1].cy + four[2].cy + four[3].cy) / 4;

  // For each square, choose the vertex pointing outward from the sheet center
  function outwardVertex(sq) {
    let best = sq.box[0], bestDot = -Infinity;
    const dirX = sq.cx - cxMean, dirY = sq.cy - cyMean;
    for (const v of sq.box) {
      const vx = v.x - sq.cx, vy = v.y - sq.cy;
      const dot = vx*dirX + vy*dirY;
      if (dot > bestDot) { bestDot = dot; best = v; }
    }
    return best; // VIDEO px
  }

  const videoCorners   = four.map(outwardVertex);
  const orderedVideo   = orderTLTRBRBLBySumDiff(videoCorners); // TL,TR,BR,BL
  const overlayCorners = orderedVideo.map(({x,y}) => videoPtToOverlayPx({x,y}));

  // Build H (overlay -> sheet)
  const srcMat = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    overlayCorners[0].px, overlayCorners[0].py,
    overlayCorners[1].px, overlayCorners[1].py,
    overlayCorners[2].px, overlayCorners[2].py,
    overlayCorners[3].px, overlayCorners[3].py,
  ]));
  const dstMat = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    0,0,  SHEET_W,0,  SHEET_W,SHEET_H,  0,SHEET_H
  ]));
  const Hmat = cv.getPerspectiveTransform(srcMat, dstMat);
  srcMat.delete(); dstMat.delete();

  if (H)    { H.delete?.();    H    = null; }
  if (Hinv) { Hinv.delete?.(); Hinv = null; }

  lastOverlayCorners = overlayCorners.slice();
  H = Hmat;

  // Build Hinv (sheet -> overlay) via swapped points (more robust than invert)
  const srcSheet   = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([ 0,0,  SHEET_W,0,  SHEET_W,SHEET_H,  0,SHEET_H ]));
  const dstOverlay = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    overlayCorners[0].px, overlayCorners[0].py,
    overlayCorners[1].px, overlayCorners[1].py,
    overlayCorners[2].px, overlayCorners[2].py,
    overlayCorners[3].px, overlayCorners[3].py,
  ]));
  Hinv = cv.getPerspectiveTransform(srcSheet, dstOverlay);
  srcSheet.delete(); dstOverlay.delete();

  // Quality gate
  const err = rmsReprojError(overlayCorners);
  if (!isFinite(err) || err > 12) {
    if (H)    { H.delete?.();    H    = null; }
    if (Hinv) { Hinv.delete?.(); Hinv = null; }
    console.warn('Calibration rejected, RMS error px =', err);
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
// Rendering: projected pads (calibrated) or simple circles (fallback)
// -----------------------------------------------------------------------------
function drawPadsProjected(ctx) {
  if (!Hinv || !cvReady) return;

  // project one sheet point -> overlay px using Hinv
  const project = (sx, sy) => {
    const src = cv.matFromArray(1, 1, cv.CV_32FC2, new Float32Array([sx, sy]));
    const dst = new cv.Mat();
    cv.perspectiveTransform(src, dst, Hinv);
    const out = { x: dst.data32F[0], y: dst.data32F[1] };
    src.delete(); dst.delete();
    return out;
  };

  const padsTL = padsForScreen();

  ctx.save();
  ctx.lineWidth   = 2;
  ctx.strokeStyle = "rgba(0,170,255,0.95)";
  ctx.fillStyle   = "rgba(0,170,255,0.95)";
  ctx.font        = "12px system-ui";
  ctx.textAlign   = "center";
  ctx.textBaseline= "middle";

  const SEG = 40; // circle approximation segments
  for (const p of padsTL) {
    ctx.beginPath();
    for (let i = 0; i <= SEG; i++) {
      const t  = (i / SEG) * Math.PI * 2;
      const sx = p.x + p.r * Math.cos(t);
      const sy = p.y + p.r * Math.sin(t);
      const { x, y } = project(sx, sy);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const c = project(p.x, p.y);
    ctx.fillText(p.name, c.x, c.y);
  }

  ctx.restore();
}

function renderOverlay(tipPx) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (Hinv && cvReady) {
    // Perspective-correct overlays
    drawPadsProjected(ctx);
  } else {
    // v1.1 overlay (simple scale)
    const pads = padsForScreen();
    const sx = overlay.width  / SHEET_W;
    const sy = overlay.height / SHEET_H;

    ctx.lineWidth = 2;
    for (const p of pads) {
      ctx.beginPath();
      ctx.arc(p.x * sx, p.y * sy, p.r * ((sx + sy) / 2), 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.stroke();

      ctx.font = "12px system-ui";
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.name, p.x * sx, p.y * sy);
    }
  }

  if (tipPx) {
    ctx.beginPath();
    ctx.arc(tipPx.px, tipPx.py, 7, 0, Math.PI*2);
    ctx.fillStyle = "rgba(0,200,255,0.95)";
    ctx.fill();
  }
}

// -----------------------------------------------------------------------------
// Main Loop
// -----------------------------------------------------------------------------
async function loop(ts) {
  if (!video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(loop);
    return;
  }

  const result = handLandmarker.detectForVideo(video, ts);

  let tipSheet = null;
  let tipPx    = null;

  if (result && result.landmarks && result.landmarks.length > 0) {
    const lm  = result.landmarks[0];
    const tip = lm[8]; // index fingertip
    const p   = tipToOverlayPx(tip.x, tip.y);

    tipPx    = p;
    tipSheet = (H && cvReady) ? mapOverlayToSheet(p.px, p.py)
                              : overlayPxToSheet(p.px, p.py);
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
          const vol = Math.min(1.0, Math.max(0.15, v / 220)); // velocity -> volume
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

    lastTip = tipSheet;
  }

  lastTime = ts;
  renderOverlay(tipPx);
  requestAnimationFrame(loop);
}

// -----------------------------------------------------------------------------
// Calibration Button / Camera Button
// -----------------------------------------------------------------------------
btnCam.onclick = async () => {
  await initCamera();
  if (!audioCtx)       await initAudio();
  if (!handLandmarker) await initHands();
  statusEl.textContent = "Camera on — show the printed sheet and tap a pad.";
  requestAnimationFrame(loop);
};

btnCal.onclick = async () => {
  statusEl.textContent = "Calibrating...";
  try {
    await ensureVideoReady();
    await loadOpenCVOnce();

    const ok = findSquaresAndHomographyFromCurrentFrame(video);
    statusEl.textContent = ok ? "Calibrated" : "Calibration failed";

    const g = overlay.getContext('2d');
    g.save();
    g.strokeStyle = ok ? "#00ff66" : "#ff3355";
    g.lineWidth   = 4;
    g.strokeRect(8,8, overlay.width-16, overlay.height-16);

    // Optional: show picked corners (quick re-run for visualization)
    if (ok) {
      const src = frameToMat(video);
      const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      const blur = new cv.Mat(); cv.GaussianBlur(gray, blur, new cv.Size(5,5), 0);
      const bin  = new cv.Mat(); cv.threshold(blur, bin, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);
      const contours = new cv.MatVector(), hierarchy = new cv.Mat();
      cv.findContours(bin, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
      const vw = src.cols, vh = src.rows, imgArea = vw*vh;
      const cand=[];
      for (let i=0;i<contours.size();i++){
        const c=contours.get(i);
        const peri=cv.arcLength(c,true), approx=new cv.Mat();
        cv.approxPolyDP(c, approx, 0.03*peri, true);
        if (approx.rows===4 && cv.isContourConvex(approx)){
          const area=cv.contourArea(approx), areaFrac=area/imgArea;
          if (areaFrac>0.0003 && areaFrac<0.25){
            const r=cv.boundingRect(approx), ar=r.width/r.height;
            if (ar>0.55 && ar<1.45){ let cx=0,cy=0;
              for (let j=0;j<4;j++){ cx+=approx.intPtr(j,0)[0]; cy+=approx.intPtr(j,0)[1]; }
              cand.push({x:cx/4, y:cy/4, area});
            }
          }
        }
        approx.delete(); c.delete();
      }
      contours.delete(); hierarchy.delete(); src.delete(); gray.delete(); blur.delete(); bin.delete();
      if (cand.length>=4){
        cand.sort((a,b)=>b.area-a.area);
        const top=cand.slice(0,8);
        let best=null,score=-1;
        for(let i=0;i<top.length;i++)
          for(let j=i+1;j<top.length;j++)
            for(let k=j+1;k<top.length;k++)
              for(let l=k+1;l<top.length;l++){
                const set=[top[i],top[j],top[k],top[l]];
                let s=0; for(let a=0;a<4;a++)for(let b=a+1;b<4;b++){
                  const dx=set[a].x-set[b].x, dy=set[a].y-set[b].y; s+=dx*dx+dy*dy; }
                if(s>score){score=s;best=set;}
              }
        if(best){
          const ordered = orderCornersTLTRBRBL(best).map(videoPtToOverlayPx);
          g.fillStyle = "#00ff66";
          for (const p of ordered) {
            g.beginPath(); g.arc(p.px,p.py,6,0,Math.PI*2); g.fill();
          }
        }
      }
    }
    g.restore();
  } catch (e) {
    console.error("Calibration error", e);
    statusEl.textContent = "Calibration error (see console)";
  }
};
