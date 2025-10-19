import { basePads } from './pads.js';
import { audioCtx, setAudioCtx, samples, cooldown, COOLDOWN_MS } from './state.js';

export async function initAudio() {
  if (!audioCtx) setAudioCtx(new (window.AudioContext || window.webkitAudioContext)());
  for (const p of basePads) {
    const ab  = await fetch(p.sound).then(r => r.arrayBuffer());
    const buf = await audioCtx.decodeAudioData(ab);
    samples.set(p.name, buf);
  }
}

export function play(name, gain=1.0) {
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
