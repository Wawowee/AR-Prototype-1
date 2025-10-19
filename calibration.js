import {
  video, SHEET_W, SHEET_H, setH, setHinv, setLastOverlayCorners,
  H, Hinv, cvReady
} from './state.js';
import { videoPtToOverlayPx } from './pads.js';

// Helpers
function frameToMat(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const c  = document.createElement('canvas');
  c.width = vw; c.height = vh;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(video, 0, 0, vw, vh);
  return cv.imread(c); // CV_8UC4
}

function rotatedRectVertices(rr) {
  if (cv.RotatedRect && typeof cv.RotatedRect.points === 'function') {
    const pts = cv.RotatedRect.points(rr);
    if (Array.isArray(pts) && pts.length === 4 && 'x' in pts[0]) return pts;
    if (pts?.data32F?.length >= 8) {
      const a = pts.data32F;
      const out = [ {x:a[0],y:a[1]}, {x:a[2],y:a[3]}, {x:a[4],y:a[5]}, {x:a[6],y:a[7]} ];
      pts.delete?.(); return out;
    }
  }
  if (typeof cv.boxPoints === 'function') {
    const m = cv.boxPoints(rr);
    if (m?.data32F?.length >= 8) {
      const a = m.data32F;
      const out = [ {x:a[0],y:a[1]}, {x:a[2],y:a[3]}, {x:a[4],y:a[5]}, {x:a[6],y:a[7]} ];
      m.delete?.(); return out;
    }
  }
  throw new Error('Could not extract rotated-rect vertices');
}

function detectSquareBoxesFullRes(src) {
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
          let cx = 0, cy = 0; for (const v of box) { cx += v.x; cy += v.y; }
          cands.push({ area, cx:cx/4, cy:cy/4, box });
        }
      }
    }
    approx.delete(); cnt.delete();
  }
  contours.delete(); hierarchy.delete(); gray.delete(); blur.delete(); bin.delete();
  return cands;
}

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

// Reprojection RMS error
function rmsReprojError(overlayPts) {
  if (!H || !Hinv || !cvReady) return Infinity;
  const src = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array(overlayPts.flatMap(p => [p.px, p.py])));
  const mid = new cv.Mat(); cv.perspectiveTransform(src, mid, H);
  const back = new cv.Mat(); cv.perspectiveTransform(mid, back, Hinv);
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const dx = overlayPts[i].px - back.data32F[2*i + 0];
    const dy = overlayPts[i].py - back.data32F[2*i + 1];
    s += dx*dx + dy*dy;
  }
  src.delete(); mid.delete(); back.delete();
  return Math.sqrt(s / 4);
}

// Public: build H/Hinv from current frame (your “rectangle hugging circles” method)
export function findSquaresAndHomographyFromCurrentFrame() {
  if (!cvReady) return false;

  const src   = frameToMat(video);
  const cands = detectSquareBoxesFullRes(src);
  src.delete();
  if (!cands || cands.length < 4) return false;

  const four = chooseFourMostSpread(cands);
  if (!four) return false;

  const left  = four.reduce((a,b) => (a.cx < b.cx ? a : b));
  const right = four.reduce((a,b) => (a.cx > b.cx ? a : b));
  const top2  = four.slice().sort((a,b) => a.cy - b.cy).slice(0, 2);
  const bot2  = four.slice().sort((a,b) => b.cy - a.cy).slice(0,  2);

  const minX = sq => Math.min(...sq.box.map(v => v.x));
  const maxX = sq => Math.max(...sq.box.map(v => v.x));
  const bottomMostY = boxes => Math.max(...boxes.flatMap(sq => sq.box.map(v => v.y)));
  const topMostY    = boxes => Math.min(...boxes.flatMap(sq => sq.box.map(v => v.y)));

  const xL   = minX(left);
  const xR   = maxX(right);
  const yTop = bottomMostY(top2);
  const yBot = topMostY(bot2);

  const overlayCorners = [
    { px: xL,   py: yTop },
    { px: xR,   py: yTop },
    { px: xR,   py: yBot },
    { px: xL,   py: yBot }
  ].map(({px,py}) => videoPtToOverlayPx({ x: px, y: py }));

  // H : overlay -> sheet
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

  if (H)    { H.delete?.(); setH(null); }
  if (Hinv) { Hinv.delete?.(); setHinv(null); }

  setLastOverlayCorners(overlayCorners);
  setH(Hmat);

  // Hinv : sheet -> overlay (swapped points)
  const srcSheet   = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([ 0,0,  SHEET_W,0,  SHEET_W,SHEET_H,  0,SHEET_H ]));
  const dstOverlay = cv.matFromArray(4,1,cv.CV_32FC2,new Float32Array([
    overlayCorners[0].px, overlayCorners[0].py,
    overlayCorners[1].px, overlayCorners[1].py,
    overlayCorners[2].px, overlayCorners[2].py,
    overlayCorners[3].px, overlayCorners[3].py,
  ]));
  const HinvMat = cv.getPerspectiveTransform(srcSheet, dstOverlay);
  srcSheet.delete(); dstOverlay.delete();
  setHinv(HinvMat);

  const err = rmsReprojError(overlayCorners);
  if (!isFinite(err) || err > 12) {
    H && H.delete?.(); setH(null);
    Hinv && Hinv.delete?.(); setHinv(null);
    console.warn('Calibration rejected, RMS error px =', err);
    return false;
  }
  return true;
}

// Public: overlay px -> sheet coords using H
export function mapOverlayToSheet(x, y) {
  if (!H || !cvReady) return null;
  const src = cv.matFromArray(1,1,cv.CV_32FC2,new Float32Array([x, y]));
  const dst = new cv.Mat();
  cv.perspectiveTransform(src, dst, H);
  const out = { x: dst.data32F[0], y: dst.data32F[1] };
  src.delete(); dst.delete();
  return out;
}
