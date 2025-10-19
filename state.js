// DOM
export const video    = document.getElementById('video');
export const overlay  = document.getElementById('overlay');
export const ctx      = overlay.getContext('2d');
export const statusEl = document.getElementById('status');
export const btnCam   = document.getElementById('btnCam');
export const btnCal   = document.getElementById('btnCal');
export const cbMirror = document.getElementById('cbMirror');

window.addEventListener("load", () => console.log("app running"));

// Sheet (keep your 6.2x4 aspect)
export const SHEET_W = 620;
export const SHEET_H = 400;

// Runtime state (audio, hands, hit-test)
export let audioCtx = null;
export const samples = new Map();
export let handLandmarker = null;

export let lastTip  = null;
export let lastTime = 0;
export const cooldown = new Map();
export const COOLDOWN_MS = 120;
export const VELOCITY_THRESH = 2.0;
export const wasInside = new Map();

// Calibration mats
export let cvReady = false;
export let H = null;       // overlay -> sheet
export let Hinv = null;    // sheet   -> overlay
export let lastOverlayCorners = null; // [{px,py} * 4]

// Safe setters (so other modules can update)
export function setAudioCtx(v) { audioCtx = v; }
export function setHandLandmarker(v) { handLandmarker = v; }
export function setCvReady(v) { cvReady = v; }
export function setH(m) { H = m; }
export function setHinv(m) { Hinv = m; }
export function setLastOverlayCorners(arr) { lastOverlayCorners = arr; }
export function setLastTip(p) { lastTip = p; }
export function setLastTime(t) { lastTime = t; }
