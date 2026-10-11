// The drone's eye, JS twin of scripts/brain/eye.py (batch 1, one ray at a time). See that file for the model.
// Panorama 72 x 12 px (5 deg/px, 360 deg x +20..-40 deg); object looks 16 x 12 px (2x2 supersampled) for the MB.
export const NC = 72, NR = 12, PX_DEG = 5.0, EL_TOP = 20.0;
export const LW = 16, LH = 12, SS = 2;
export const ACUITY_DEG = 2.0, DETECT_DEG = 4.0, NOISE = 0.03, DEPTH_MAX = 20.0, FAR = 1e9;
export const PATTERNS = { stripes: 0, chevron: 1, dots: 2 };
const DOOR_RGB = [0.12, 0.10, 0.08], FOG_RGB = [0.80, 0.82, 0.85];
const D2R = Math.PI / 180;
const HZ = [0.75, 0.85, 0.95], ZEN = [0.35, 0.55, 0.90], SUNC = [1.0, 1.0, 0.9];
const COS6 = Math.cos(6.0 * D2R);

/** Python-style float modulo (sign of the divisor), as numpy's % . */
export const pmod = (x, m) => { const r = x % m; return (r !== 0 && (r < 0) !== (m < 0)) ? r + m : r; };
export const wrap = (a) => pmod(a + Math.PI, 2 * Math.PI) - Math.PI;
export const colAz = Array.from({ length: NC }, (_, c) => (180.0 - (c + 0.5) * PX_DEG) * D2R);
export const rowEl = Array.from({ length: NR }, (_, r) => (EL_TOP - (r + 0.5) * PX_DEG) * D2R);
const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);
const norm3 = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

/** Scene arrays for one world (C5 dict). */
export class Scene {
  constructor(w) {
    this.cyl = w.obstacles.filter((o) => o.type !== 'building').map((o) => [o.x, o.y, o.r, o.h, ...o.rgb]);
    this.box = w.obstacles.filter((o) => o.type === 'building').map((o) => [o.x, o.y, o.w / 2, o.d / 2, o.h, ...o.rgb]);
    this.K = w.objects.length;
    this.obj = [];
    this.role = [];
    for (const o of w.objects) {
      const a = o.appearance;
      this.obj[o.slot] = [o.x, o.y, o.z, o.yaw_deg * D2R, o.dims[0] / 2, o.dims[1] / 2, o.dims[2], ...a.base, ...a.accent,
        o.role === 'target' ? 1 : 0, PATTERNS[a.pattern], a.period_m, o.t_s];
      this.role[o.slot] = o.role === 'target' ? 1 : 0;
    }
    this.ground = w.ground.rgb;
    const az = w.sun.azimuth_deg * D2R, el = w.sun.elevation_deg * D2R;
    this.sunAz = az; this.sunEl = el;
    this.sun = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    this.light = w.light;
    this.fog = w.fog_m ? w.fog_m : 0.0;
  }
}

// ------------------------------------------------------------ primitives, one ray: o [3], d [3]
export function hitCyl(o, d, cyl) {
  let best = FAR, bi = 0, bTop = false;
  for (let i = 0; i < cyl.length; i++) {
    const [cx, cy, r, h] = cyl[i];
    const px = o[0] - cx, py = o[1] - cy;
    const a = d[0] * d[0] + d[1] * d[1];
    const bq = px * d[0] + py * d[1];
    const c = px * px + py * py - r * r;
    const disc = bq * bq - a * c;
    const sq = Math.sqrt(Math.max(disc, 0.0));
    const ts = (-bq - sq) / a;
    const zs = o[2] + ts * d[2];
    const side = disc > 0 && a > 1e-12 && ts > 1e-6 && zs >= 0 && zs <= h;
    const tt = (h - o[2]) / d[2];
    const qx = px + tt * d[0], qy = py + tt * d[1];
    const top = d[2] < 0 && o[2] > h && tt > 1e-6 && qx * qx + qy * qy <= r * r;
    const t = side ? ts : top ? tt : FAR;
    if (t < best) { best = t; bi = i; bTop = top && !side; }
  }
  let n = [0, 0, 1];
  if (!bTop && cyl.length) {
    const hx = o[0] + best * d[0] - cyl[bi][0], hy = o[1] + best * d[1] - cyl[bi][1];
    const nr = Math.sqrt(hx * hx + hy * hy) + 1e-12;
    n = [hx / nr, hy / nr, 0];
  }
  return [best, bi, n];
}

function slab(o, d, lo, hi) {
  let tnear = -Infinity, tfar = Infinity, axis = 0;
  for (let k = 0; k < 3; k++) {
    const inv = 1.0 / (Math.abs(d[k]) < 1e-12 ? 1e-12 : d[k]);
    const t1 = (lo[k] - o[k]) * inv, t2 = (hi[k] - o[k]) * inv;
    const tn = Math.min(t1, t2), tf = Math.max(t1, t2);
    if (tn > tnear) { tnear = tn; axis = k; }
    if (tf < tfar) tfar = tf;
  }
  return [tnear, axis, tnear <= tfar && tnear > 1e-6];
}

export function hitBox(o, d, box) {
  let best = FAR, bi = 0, ba = 0;
  for (let i = 0; i < box.length; i++) {
    const c = box[i];
    const [tn, ax, ok] = slab(o, d, [c[0] - c[2], c[1] - c[3], 0], [c[0] + c[2], c[1] + c[3], c[4]]);
    const t = ok ? tn : FAR;
    if (t < best) { best = t; bi = i; ba = ax; }
  }
  const n = [0, 0, 0];
  n[ba] = -sign(d[ba]);
  return [best, bi, n];
}

/** objects: array of 17-vectors (null = absent slot). -> [t, idx, normal, rgb] */
export function hitObj(o, d, objs, tNow) {
  let best = FAR, res = null;
  for (let i = 0; i < objs.length; i++) {
    const c = objs[i];
    if (!c || !(c[16] <= tNow && c[0] < FAR / 2)) continue;
    const cs = Math.cos(c[3]), sn = Math.sin(c[3]);
    const rx = o[0] - c[0], ry = o[1] - c[1];
    const lo = [cs * rx + sn * ry, -sn * rx + cs * ry, o[2] - c[2]];
    const ld = [cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1], d[2] + 0.0 * cs];
    const [tn, a, ok] = slab(lo, ld, [-c[4], -c[5], 0], [c[4], c[5], c[6]]);
    if (ok && tn < best) { best = tn; res = [i, a, lo, ld]; }
  }
  if (!res) return [FAR, 0, [0, 0, 0], [0, 0, 0]];
  const [i, a, lo, ld] = res, c = objs[i];
  const P = [lo[0] + best * ld[0], lo[1] + best * ld[1], lo[2] + best * ld[2]];
  const hd = c[5], hh = c[6];
  const u = a === 0 ? P[1] : P[0];
  const v = a === 2 ? P[1] : P[2];
  const per = c[15], pat = c[14];
  const fu = u / per - Math.floor(u / per), fv = v / per - Math.floor(v / per);
  const fc = (Math.abs(u) + v) / per;
  const on = pat === 0 ? fu < 0.5 : pat === 1 ? fc - Math.floor(fc) < 0.5 : (fu - 0.5) ** 2 + (fv - 0.5) ** 2 < 0.09;
  let rgb = on ? [c[10], c[11], c[12]] : [c[7], c[8], c[9]];
  const front = a === 0 && ld[0] < 0 && Math.abs(P[1]) < 0.2 * hd * 2 && P[2] < 0.75 * hh;
  if (front) rgb = DOOR_RGB.slice();
  const nl = [0, 0, 0];
  nl[a] = -sign(ld[a]);
  const cy = Math.cos(c[3]), sy = Math.sin(c[3]);
  return [best, i, [cy * nl[0] - sy * nl[1], sy * nl[0] + cy * nl[1], nl[2]], rgb];
}

/** One ray, full colour. -> [rgb [3] in 0..1, dist] */
export function shade(sc, o, d, tNow, { objects = true, obstacles = true, objs = null, falseSun = null } = {}) {
  const lam = (n) => 0.45 + 0.55 * Math.max(0.0, n[0] * sc.sun[0] + n[1] * sc.sun[1] + n[2] * sc.sun[2]);
  let best = d[2] < -1e-9 ? -o[2] / d[2] : FAR;
  const gl = 0.45 + 0.55 * Math.sin(sc.sunEl);
  let rgb = [sc.ground[0] * gl, sc.ground[1] * gl, sc.ground[2] * gl];
  if (obstacles) {
    const [tc, ic, nc] = hitCyl(o, d, sc.cyl);
    if (tc < best) { const l = lam(nc), c = sc.cyl[ic]; rgb = [c[4] * l, c[5] * l, c[6] * l]; best = tc; }
    const [tb, ib, nb] = hitBox(o, d, sc.box);
    if (tb < best) { const l = lam(nb), c = sc.box[ib]; rgb = [c[5] * l, c[6] * l, c[7] * l]; best = tb; }
  }
  if (objects) {
    const [to, , no, co] = hitObj(o, d, objs || sc.obj, tNow);
    if (to < best) { const l = lam(no); rgb = [co[0] * l, co[1] * l, co[2] * l]; best = to; }
  }
  const el = Math.asin(Math.min(Math.max(d[2], -1), 1));
  const f = Math.min(Math.max(el / (Math.PI / 2), 0), 1);
  let sky = [HZ[0] + (ZEN[0] - HZ[0]) * f, HZ[1] + (ZEN[1] - HZ[1]) * f, HZ[2] + (ZEN[2] - HZ[2]) * f];
  const cosang = d[0] * sc.sun[0] + d[1] * sc.sun[1] + d[2] * sc.sun[2];
  if (cosang > COS6) sky = sky.map((s, k) => s + 1.0 * (SUNC[k] - s));
  if (falseSun !== null && Number.isFinite(falseSun)) {
    const ce = Math.cos(sc.sunEl);
    const fs = [ce * Math.cos(falseSun), ce * Math.sin(falseSun), Math.sin(sc.sunEl)];
    if (d[0] * fs[0] + d[1] * fs[1] + d[2] * fs[2] > COS6) sky = sky.map((s, k) => s + 1.0 * (SUNC[k] - s));
  }
  const hit = best < FAR / 2;
  if (!hit) rgb = sky;
  const fogf = sc.fog > 0 && hit ? Math.exp(-best / sc.fog) : 1.0;
  rgb = rgb.map((v, k) => Math.min(Math.max((v * fogf + FOG_RGB[k] * (1 - fogf)) * sc.light, 0), 1));
  return [rgb, hit ? best : FAR];
}

/** -> Uint8Array(NR*NC*3) row-major; dead: bool[NC] or null; falseSun: azimuth rad or null. */
export function panorama(sc, pos, yaw, tNow, dead = null, falseSun = null) {
  const img = new Uint8Array(NR * NC * 3);
  for (let r = 0; r < NR; r++) {
    const el = rowEl[r], ce = Math.cos(el), se = Math.sin(el);
    for (let c = 0; c < NC; c++) {
      if (dead && dead[c]) continue;
      const az = yaw + colAz[c];
      const [rgb] = shade(sc, pos, [ce * Math.cos(az), ce * Math.sin(az), se], tNow, { falseSun });
      for (let k = 0; k < 3; k++) img[(r * NC + c) * 3 + k] = Math.floor(rgb[k] * 255 + 0.5);
    }
  }
  return img;
}

export const objCentre = (c) => [c[0], c[1], c[2] + c[6] / 2];
export const objDiag = (c) => 2 * Math.sqrt(c[4] ** 2 + c[5] ** 2 + (c[6] / 2) ** 2);

function blurMat(sigma, n) {
  const s = Math.max(sigma, 1e-3), G = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n);
    let tot = 0;
    for (let j = 0; j < n; j++) { row[j] = Math.exp(-0.5 * ((i - j) / s) ** 2); tot += row[j]; }
    for (let j = 0; j < n; j++) row[j] /= tot;
    G.push(row);
  }
  return G;
}

/** Look of object slot k from eye position e. -> [grey Float64Array(LH*LW) (pre-noise), range]. */
export function look(sc, k, e, tNow) {
  const c = sc.obj[k];
  const ctr = objCentre(c);
  let f = [ctr[0] - e[0], ctr[1] - e[1], ctr[2] - e[2]];
  const rng = norm3(f);
  const fr = Math.max(rng, 1e-9);
  f = [f[0] / fr, f[1] / fr, f[2] / fr];
  let rt = [f[1], -f[0], 0];
  const rn = Math.max(norm3(rt), 1e-9);
  rt = [rt[0] / rn, rt[1] / rn, 0];
  const up = [rt[1] * f[2] - rt[2] * f[1], rt[2] * f[0] - rt[0] * f[2], rt[0] * f[1] - rt[1] * f[0]];
  const win = 2 * Math.atan(0.8 * objDiag(c) / Math.max(rng, 1e-3));
  const W = LW * SS, H = LH * SS, pitch = win / W;
  const grey = [];
  const objs = [c];
  for (let j = 0; j < H; j++) {
    const row = new Float64Array(W);
    const ty = Math.tan(((H - 1) / 2 - j) * pitch);
    for (let i = 0; i < W; i++) {
      const tx = Math.tan((i - (W - 1) / 2) * pitch);
      let d = [f[0] + tx * rt[0] + ty * up[0], f[1] + tx * rt[1] + ty * up[1], f[2] + tx * rt[2] + ty * up[2]];
      const dn = norm3(d);
      d = [d[0] / dn, d[1] / dn, d[2] / dn];
      const [rgb] = shade(sc, e, d, tNow, { obstacles: false, objs });
      row[i] = (rgb[0] + rgb[1] + rgb[2]) / 3;
    }
    grey.push(row);
  }
  const sig = (ACUITY_DEG * D2R) / pitch;
  const gx = blurMat(sig, W), gy = blurMat(sig, H);
  // (gy @ grey) @ gx^T, the same association as eye.py
  const tmp = [];
  for (let j = 0; j < H; j++) { const o = new Float64Array(W); for (let i = 0; i < W; i++) { let s = 0; for (let m = 0; m < H; m++) s += gy[j][m] * grey[m][i]; o[i] = s; } tmp.push(o); }
  const bl = tmp.map((row) => { const o = new Float64Array(W); for (let i = 0; i < W; i++) { let s = 0; for (let m = 0; m < W; m++) s += row[m] * gx[i][m]; o[i] = s; } return o; });
  const out = new Float64Array(LH * LW);
  for (let y = 0; y < LH; y++) for (let x = 0; x < LW; x++)
    out[y * LW + x] = (bl[2 * y][2 * x] + bl[2 * y][2 * x + 1] + bl[2 * y + 1][2 * x] + bl[2 * y + 1][2 * x + 1]) / 4;
  return [out, rng];
}

/** -> {vis: bool[K], bearing: [K] (world rad), rh: [K]} */
export function visibility(sc, pos, yaw, tNow, dead = null, occluded = null) {
  const K = sc.K, vis = [], bearing = [], rh = [];
  const tanDet = Math.tan(DETECT_DEG * D2R);
  for (let k = 0; k < K; k++) {
    const c = sc.obj[k], ctr = objCentre(c);
    const rel = [ctr[0] - pos[0], ctr[1] - pos[1], ctr[2] - pos[2]];
    bearing.push(Math.atan2(rel[1], rel[0]));
    rh.push(Math.hypot(rel[0], rel[1]));
    const dist = norm3(rel);
    let ok = c[16] <= tNow && objDiag(c) / Math.max(dist, 1e-6) >= tanDet;
    if (sc.fog > 0) ok = ok && Math.exp(-dist / sc.fog) >= 0.2;
    let clear = 0;
    for (const s of [0.0, 0.8, -0.8]) {
      const P = [ctr[0] + s * c[4] * Math.cos(c[3]), ctr[1] + s * c[4] * Math.sin(c[3]), ctr[2]];
      const v = [P[0] - pos[0], P[1] - pos[1], P[2] - pos[2]];
      const L = norm3(v), Lm = Math.max(L, 1e-9);
      const d = [v[0] / Lm, v[1] / Lm, v[2] / Lm];
      if (Math.min(hitCyl(pos, d, sc.cyl)[0], hitBox(pos, d, sc.box)[0]) >= L) clear++;
    }
    ok = ok && clear >= 2;
    if (dead) {
      const relAz = wrap(bearing[k] - yaw);
      const col = Math.min(Math.max(Math.floor((Math.PI - relAz) / (PX_DEG * D2R)), 0), NC - 1);
      ok = ok && !dead[col];
    }
    if (occluded) ok = ok && !occluded[k];
    vis.push(ok);
  }
  return { vis, bearing, rh };
}

export function depthRing(sc, pos, yaw) {
  const out = new Float64Array(NC);
  for (let c = 0; c < NC; c++) {
    const az = yaw + colAz[c];
    const d = [Math.cos(az), Math.sin(az), 0];
    out[c] = Math.min(Math.min(hitCyl(pos, d, sc.cyl)[0], hitBox(pos, d, sc.box)[0]), DEPTH_MAX);
  }
  return out;
}

export function learningPoses(sc, learningFlightS) {
  const V = Math.max(1, Math.floor(5 * learningFlightS + 0.5));
  const T = sc.obj[sc.role.indexOf(1)];
  const out = [];
  for (let v = 0; v < V; v++) {
    let az, rad, alt;
    if (V === 1) { az = 0; rad = 4.0; alt = 2.5; } else {
      const p = v / (V - 1);
      az = ((40 + 80 * p) * D2R) * Math.sin(3 * Math.PI * p);
      rad = 2 + 5 * p; alt = 1 + 2.5 * p;
    }
    const a = T[3] + az;
    out.push([T[0] + rad * Math.cos(a), T[1] + rad * Math.sin(a), alt]);
  }
  return out;
}

export function learningLooks(sc, learningFlightS) {
  const k = sc.role.indexOf(1);
  return learningPoses(sc, learningFlightS).map((p) => look(sc, k, p, 1e9)[0]);
}

/** n sequential Rng.normal() draws (same order as eye.py normals()). */
export function normals(rng, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.normal();
  return out;
}
