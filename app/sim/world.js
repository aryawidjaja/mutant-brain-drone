// World builder, JS twin of scripts/world.py. Deterministic from scenario.seed via Rng stream 0.
// Angles CCW from +x (east), including yaw_deg and sun.azimuth_deg. parity_brain.py asserts equality with Python.
import { Rng } from './rng.js';

export const N_MAX = { meadow: 40, forest: 250, urban: 60 };
export const GROUND = { meadow: [0.36, 0.52, 0.24], forest: [0.28, 0.33, 0.18], urban: [0.42, 0.42, 0.44] };
export const DIMS = { tent: [2.2, 1.8, 1.3], pad: [2.0, 2.0, 0.12], backpack: [0.6, 0.4, 0.7] };
export const GALLERY = {
  tent: [[[0.90, 0.45, 0.10], [0.95, 0.90, 0.80], 'stripes', 0.50], [[0.15, 0.45, 0.80], [0.95, 0.85, 0.20], 'chevron', 0.60],
    [[0.80, 0.15, 0.15], [0.95, 0.95, 0.95], 'dots', 0.45], [[0.20, 0.60, 0.30], [0.95, 0.75, 0.10], 'stripes', 0.40]],
  pad: [[[0.95, 0.85, 0.10], [0.10, 0.10, 0.10], 'chevron', 0.50], [[0.90, 0.30, 0.10], [0.95, 0.95, 0.95], 'stripes', 0.40],
    [[0.20, 0.30, 0.85], [0.95, 0.95, 0.95], 'dots', 0.40], [[0.95, 0.95, 0.95], [0.85, 0.10, 0.10], 'chevron', 0.60]],
  backpack: [[[0.85, 0.20, 0.10], [0.15, 0.15, 0.15], 'stripes', 0.15], [[0.10, 0.35, 0.75], [0.95, 0.80, 0.10], 'dots', 0.15],
    [[0.95, 0.60, 0.05], [0.20, 0.20, 0.20], 'chevron', 0.20], [[0.30, 0.55, 0.20], [0.90, 0.90, 0.85], 'stripes', 0.12]],
};
export const CLEAR_M = 3.0;
// v2 only: the home landing pad (see scripts/world.py HOME_PAD); not an object slot.
export const HOME_PAD = { role: 'home', kind: 'home_pad', yaw_deg: 0.0, dims: [1.0, 1.0, 0.06], t_s: 0.0, slot: 0,
  appearance: { base: [0.75, 0.10, 0.60], accent: [0.97, 0.97, 0.97], pattern: 'stripes', period_m: 0.25, similarity: 1.0 } };

const lerp = (a, b, s) => a.map((x, i) => x + (b[i] - x) * s);

function boxDist(px, py, o) {
  const dx = Math.max(Math.abs(px - o.x) - o.w / 2, 0.0);
  const dy = Math.max(Math.abs(py - o.y) - o.d / 2, 0.0);
  return (dx * dx + dy * dy) ** 0.5;
}

/** scenario (validated, defaults filled) -> world dict. */
export function buildWorld(sc) {
  const r = new Rng(sc.seed, 0);
  const U = () => r.uniform();
  const [W, H] = sc.world.size_m;
  const land = sc.world.landscape;
  const sun = { azimuth_deg: 360 * U(), elevation_deg: 25 + 35 * U() };
  const t = sc.target;
  const u = U();   // always drawn, so a chosen picture leaves the layout unchanged
  const [base, accent, pattern, period] = GALLERY[t.kind][t.picture ?? Math.min(3, Math.floor(u * 4))];
  let objects = [{ role: 'target', kind: t.kind, x: t.pos[0], y: t.pos[1], z: t.pos[2], yaw_deg: t.yaw_deg, dims: DIMS[t.kind], t_s: 0.0,
    appearance: { base, accent, pattern, period_m: period, similarity: 1.0 } }];
  for (const dc of sc.decoys) {
    const s = dc.similarity;
    const offB = [U(), U(), U()], offA = [U(), U(), U()];
    const same = dc.kind === t.kind;
    const [db, da, dp, dper] = same ? [base, accent, pattern, period] : GALLERY[dc.kind][0];
    objects.push({ role: 'decoy', kind: dc.kind, x: dc.pos[0], y: dc.pos[1], z: dc.pos[2], yaw_deg: t.yaw_deg + 360 * U(),
      dims: DIMS[dc.kind], t_s: dc.t_s,
      appearance: { base: lerp(offB, db, s), accent: lerp(offA, da, s), pattern: dp, period_m: dper * (1 + 0.8 * (1 - s)), similarity: s } });
  }
  const order = objects.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor(U() * (i + 1)));
    [order[i], order[j]] = [order[j], order[i]];
  }
  objects = order.map((k) => objects[k]);
  objects.forEach((o, k) => { o.slot = k; });
  const clear = [[sc.home[0], sc.home[1]], ...objects.map((o) => [o.x, o.y])];
  const n = Math.floor(sc.world.obstacle_density * N_MAX[land] + 0.5);
  const obstacles = [];
  for (let i = 0; i < n; i++) {
    for (let tr = 0; tr < 30; tr++) {
      const x = (U() - 0.5) * W, y = (U() - 0.5) * H;
      const kindU = U(), a = U(), b = U(), c = U(), g = U();
      let o, ok;
      if (land === 'urban') {
        o = { type: 'building', x, y, w: 4 + 8 * a, d: 4 + 8 * b, h: 5 + 10 * c, rgb: [0.55 + 0.25 * g, 0.53 + 0.22 * g, 0.50 + 0.2 * g] };
        ok = clear.every(([px, py]) => boxDist(px, py, o) > CLEAR_M);
      } else {
        const tree = kindU < (land === 'forest' ? 0.9 : 0.35);
        o = tree ? { type: 'tree', x, y, r: 0.8 + 0.8 * a, h: 4 + 5 * b, rgb: [0.10 + 0.10 * g, 0.30 + 0.20 * c, 0.08 + 0.08 * g] }
          : { type: 'rock', x, y, r: 0.3 + 0.5 * a, h: 0.4 + 0.8 * b, rgb: [0.45 + 0.2 * g, 0.45 + 0.2 * g, 0.43 + 0.2 * g] };
        ok = clear.every(([px, py]) => ((px - x) ** 2 + (py - y) ** 2) ** 0.5 > CLEAR_M + o.r);
      }
      if (ok) { obstacles.push(o); break; }
    }
  }
  const w = { size: [W, H], ground: { rgb: GROUND[land] }, obstacles, objects, home: sc.home.slice(), sun,
    light: sc.world.light, fog_m: sc.world.fog_m };
  if ((sc.brain.version ?? 1) === 2) w.home_pad = { ...JSON.parse(JSON.stringify(HOME_PAD)), x: sc.home[0], y: sc.home[1], z: sc.home[2] };
  return w;
}
