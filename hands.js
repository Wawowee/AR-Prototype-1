import { setHandLandmarker } from './state.js';
const MP_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

export async function initHands() {
  const { FilesetResolver, HandLandmarker } = await import(`${MP_URL}`);
  const filesetResolver = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
  const lm = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    numHands: 1,
    runningMode: "VIDEO"
  });
  setHandLandmarker(lm);
}
