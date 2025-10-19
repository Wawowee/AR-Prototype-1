import { SHEET_W, SHEET_H, ctx, overlay, cvReady, Hinv } from './state.js';
import { padsForScreen } from './pads.js';

export function drawPadsProjected() {
  if (!Hinv || !cvReady) return;

  const project = (sx, sy) => {
    const src = cv.matFromArray(1,1,cv.CV_32FC2,new Float32Array([sx, sy]));
    const dst = new cv.Mat(); cv.perspectiveTransform(src, dst, Hinv);
    const out = { x: dst.data32F[0], y: dst.data32F[1] };
    src.delete(); dst.delete();
    return out;
    };

  const padsTL = padsForScreen();
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(0,170,255,0.95)";
  ctx.fillStyle   = "rgba(0,170,255,0.95)";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const SEG = 40;
  for (const p of padsTL) {
    ctx.beginPath();
    for (let i = 0; i <= SEG; i++) {
      const t = (i / SEG) * Math.PI * 2;
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

export function renderOverlay(tipPx) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (Hinv && cvReady) {
    drawPadsProjected();
  } else {
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
