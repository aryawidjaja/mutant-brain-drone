// Obstacle reflexes (flow balancing + looming), JS twin of scripts/brain/avoid.py. Added after the arbiter at the
// descending-neuron level (Rayshubskiy et al. 2025, eLife 102230). Input: geometric optic flow from ray-casts plus noise.
import { colAz, NC } from './eye.js';

export const K_BAL = 0.6, F_REF = 0.5, TAU_LOOM = 2.0, K_LOOM = 1.2, SIGMA_FLOW = 0.05, R_DRONE = 0.15;
const D2R = Math.PI / 180;
const PHI = colAz;
const LEFT = PHI.map((p) => p >= 20 * D2R && p <= 90 * D2R);
const RIGHT = PHI.map((p) => p <= -20 * D2R && p >= -90 * D2R);
const FRONT = PHI.map((p) => Math.abs(p) <= 30 * D2R);
const FL = PHI.map((p) => p > 0 && p <= 90 * D2R);
const FR = PHI.map((p) => p < 0 && p >= -90 * D2R);
const mean = (a, m) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) if (m[i]) { s += a[i]; n++; } return s / n; };

/** depth [NC] m, vBody [f, l], noise [NC] normals, dead bool[NC] or null -> {yawOffset, speedFactor, loom} */
export function reflex(depth, vBody, noise, dead = null) {
  const [vf, vs] = vBody;
  const F = new Float64Array(NC), E = new Float64Array(NC), D = Float64Array.from(depth);
  for (let i = 0; i < NC; i++) {
    F[i] = Math.abs(vf * Math.sin(PHI[i]) - vs * Math.cos(PHI[i])) / depth[i] + SIGMA_FLOW * noise[i];
    E[i] = Math.max(0.0, vf * Math.cos(PHI[i]) + vs * Math.sin(PHI[i])) / depth[i];
    if (dead && dead[i]) { F[i] = 0.0; E[i] = 0.0; D[i] = 20.0; }
  }
  const fl = mean(F, LEFT), fr = mean(F, RIGHT);
  const bal = -K_BAL * (fl - fr) / (fl + fr + 1e-6) * Math.tanh(Math.max(fl, fr) / F_REF);
  let loom = -Infinity;
  for (let i = 0; i < NC; i++) if (FRONT[i] && E[i] > loom) loom = E[i];
  const excess = Math.min(Math.max(loom * TAU_LOOM - 1.0, 0.0), 1.0);
  const side = mean(D, FL) >= mean(D, FR) ? 1.0 : -1.0;
  return { yawOffset: bal + side * K_LOOM * excess, speedFactor: 1.0 - 0.7 * excess, loom };
}

/** true if the drone sphere is inside an obstacle. sc: eye.Scene */
export function collided(sc, pos) {
  for (const c of sc.cyl) if (Math.hypot(pos[0] - c[0], pos[1] - c[1]) < c[2] + R_DRONE && pos[2] < c[3] + R_DRONE) return true;
  for (const b of sc.box)
    if (Math.abs(pos[0] - b[0]) < b[2] + R_DRONE && Math.abs(pos[1] - b[1]) < b[3] + R_DRONE && pos[2] < b[4] + R_DRONE) return true;
  return false;
}
