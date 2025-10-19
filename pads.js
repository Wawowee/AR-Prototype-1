import { SHEET_W, SHEET_H, overlay, video, cbMirror } from './state.js';

// PDF geometry (fractions)
const PAD_SCALE = 1;                       // keep your tweak
const X_FRACS = [5/24, 3/6, 5/6];          // your final tuned columns
const Y_TOP_FRAC = 0.7125;                 // BL-origin (top row)
const Y_BOT_FRAC = 0.2875;                 // BL-origin (bottom row)
const R_FRAC     = 0.85 / 6.2;             // radius as fraction of width

// Micro alignment
const TOP_ROW_DY = -22;
const BOT_ROW_DY = 0;
const COL_DX     = [-8, 0, 0];

// Names + sounds only
export const basePads = [
  { name: "Kick",    sound: "sounds/kick.wav" },
  { name: "Snare",   sound: "sounds/snare.wav" },
  { name: "HiHat C", sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     sound: "sounds/tom.wav" },
  { name: "Clap",    sound: "sounds/clap.wav" },
  { name: "HiHat O", sound: "sounds/hihat_open.wav" },
];

// Map names → (column index, row)
const PAD_INDEX = {
  "Tom":     [0, "top"],
  "Clap":    [1, "top"],
  "HiHat O": [2, "top"],
  "Kick":    [0, "bot"],
  "Snare":   [1, "bot"],
  "HiHat C": [2, "bot"],
};

// Single source of truth (top-left sheet coords)
export function padsForScreen() {
  const r = Math.round(SHEET_W * R_FRAC * PAD_SCALE);
  return basePads.map(p => {
    const [ci, row] = PAD_INDEX[p.name];
    const x = Math.round(SHEET_W * X_FRACS[ci]) + COL_DX[ci];
    const yBL = Math.round(SHEET_H * (row === "top" ? Y_TOP_FRAC : Y_BOT_FRAC));
    let yTL = SHEET_H - yBL;
    yTL += (row === "top" ? TOP_ROW_DY : BOT_ROW_DY);
    return { ...p, x, y: yTL, r };
  });
}

// ---------- overlay/video mapping helpers (unchanged logic) ----------
function getCoverMapping(overlayW, overlayH, videoW, videoH) {
  const scale    = Math.max(overlayW / videoW, overlayH / videoH);
  const displayW = videoW * scale;
  const displayH = videoH * scale;
  const offsetX  = (overlayW - displayW) / 2;
  const offsetY  = (overlayH - displayH) / 2;
  return { displayW, displayH, offsetX, offsetY };
}

export function tipToOverlayPx(tipNormX, tipNormY) {
  const overlayW = overlay.width;
  const overlayH = overlay.height;
  const videoW   = video.videoWidth || overlayW;
  const videoH   = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - tipNormX) : tipNormX;
  const ny = tipNormY;

  return { px: offsetX + nx * displayW, py: offsetY + ny * displayH };
}

export function overlayPxToSheet(px, py) {
  const u = px / overlay.width;
  const v = py / overlay.height;
  return { x: u * SHEET_W, y: v * SHEET_H };
}

export function videoPtToOverlayPx({x, y}) {
  const overlayW = overlay.width, overlayH = overlay.height;
  const videoW   = video.videoWidth || overlayW;
  const videoH   = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - x / videoW) : (x / videoW);
  const ny = y / videoH;

  return { px: offsetX + nx * displayW, py: offsetY + ny * displayH };
}
