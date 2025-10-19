import { setCvReady } from './state.js';

let cvLoadPromise = null;

export async function loadOpenCVOnce() {
  if (window.cv && typeof window.cv.Mat === 'function') { setCvReady(true); return; }
  if (cvLoadPromise) return cvLoadPromise;

  const sources = [
    'https://docs.opencv.org/4.x/opencv.js',
    'https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/platforms/js/opencv.js'
  ];

  cvLoadPromise = new Promise((resolve, reject) => {
    let idx = 0;
    const tryNext = () => {
      if (idx >= sources.length) { reject(new Error('Failed to load OpenCV.js')); return; }
      const url = sources[idx++];
      const s = document.createElement('script');
      s.src = url; s.async = true;
      s.onload = () => {
        const wait = () => {
          const ok = window.cv && typeof cv.Mat === 'function' && typeof cv.getPerspectiveTransform === 'function';
          if (ok) { setCvReady(true); resolve(); } else setTimeout(wait, 25);
        };
        if (window.cv && typeof cv.onRuntimeInitialized === 'function') {
          cv.onRuntimeInitialized = () => { setCvReady(true); resolve(); };
          wait();
        } else wait();
      };
      s.onerror = () => { console.warn('OpenCV load failed from', url); tryNext(); };
      document.head.appendChild(s);
    };
    tryNext();
  });

  return cvLoadPromise;
}
