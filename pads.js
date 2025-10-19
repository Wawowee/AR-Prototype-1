// app.js
// Minimal demo using utils. Works in a browser with <script type="module">,
// or in Node 18+ with `node --experimental-modules` or native ESM.

import { clamp, lerp, debounce, randomInt, toRadians } from './utils.js';

// Simple demo: animate a value in the console between 0 and 100.
let t = 0;
let dir = 1;

const tick = () => {
  t += 0.02 * dir;
  if (t >= 1 || t <= 0) dir *= -1;
  const value = Math.round(lerp(0, 100, t));
  const clamped = clamp(value, 10, 90);
  const angle = toRadians(value);
  console.log(`value=${value} clamped=${clamped} rand=${randomInt(1,6)} angle(rad)=${angle.toFixed(2)}`);
};

// Use debounce to limit how often we log a "resized" message (browser only).
const onResize = debounce(() => console.log('resized!'), 250);
if (typeof window !== 'undefined') {
  window.addEventListener('resize', onResize);
}

// Start a tiny loop (works in Node and browser). Stop after ~3 seconds.
const interval = setInterval(tick, 50);
setTimeout(() => {
  clearInterval(interval);
  console.log('Demo done.');
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', onResize);
  }
}, 3000);

// Export something just to show dual usage.
export const demoRunning = true;
