// Adaptive search (post-study module "adaptive_search", brain.version 2 only), JS twin of scripts/brain/search.py
// (formulas and rationale in its docstring). Scalar versions for one mission.
export const DETECT_DEG = 4.0, Q_DEG = 2.0, DET_MARGIN_DEG = 2.0, EL_LOW_DEG = -40.0, EL_MARGIN_DEG = 5.0;
const RAD = Math.PI / 180;                       // as Python's math.radians: x * (pi / 180)
export const EL_USE = (-EL_LOW_DEG - EL_MARGIN_DEG) * RAD;
export const ALT_MIN = 1.0, D_PRIOR = 1.0, H_PRIOR = 0.7;
export const FRONT_HALF_M = 1.1, D_REF = Math.sqrt(2.2 * 2.2 + 1.8 * 1.8 + 1.3 * 1.3);
export const N_CAL = 8, NEAR = [1.5, 4.0], FAR_FRAC = [0.75, 1.0], FAR_CAP = 20.0, MIN_SPAN = 0.01;
export const DIP_ON = 0.15, DIP_OFF = 0.05;

const q = (a) => { const s = Q_DEG * RAD; return s * Math.floor(a / s + 0.5); };
/** numpy median (mean of the two middle values for an even count). */
export function median(a) {
  const s = Array.from(a).sort((x, y) => x - y), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** poses [[x, y, z]...] of the learning flight; target base (tx, ty), height h, diagonal diag -> {D, H}. */
export function sizeEstimate(poses, tx, ty, h, diag) {
  const D = [], H = [];
  for (const [px, py, z] of poses) {
    const dx = px - tx, dy = py - ty, d = Math.sqrt(dx * dx + dy * dy);
    const eTop = Math.atan2(h - z, d), eBot = Math.atan2(-z, d);
    const zc = z - h / 2;
    const alpha = Math.atan(diag / Math.sqrt(d * d + zc * zc));
    const Hk = d * (Math.tan(q(eTop)) - Math.tan(q(eBot)));
    const zr = z - Hk / 2;
    H.push(Hk); D.push(Math.sqrt(d * d + zr * zr) * Math.tan(q(alpha)));
  }
  return { D: median(D), H: median(H) };
}

/** -> {h: search altitude, p: spiral pitch, rDip: horizontal dip range}. */
export function geometry(D, H, pStudy, altMax) {
  const R = D / Math.tan((DETECT_DEG + DET_MARGIN_DEG) * RAD);
  const p = Math.min(pStudy, 2 * R * Math.cos(EL_USE));
  const zc = H / 2;
  const h = Math.min(Math.max(zc + p / 2 * Math.tan(EL_USE), ALT_MIN), altMax);
  return { h, p, rDip: Math.sqrt(Math.max(R * R - (h - zc) * (h - zc), 0.0)) };
}

export const halfM = (D) => FRONT_HALF_M * Math.min(1.0, D / D_REF);

/** look-back views (not stored): N_CAL near then N_CAL far, azimuths 2 pi j / N_CAL, altitude h. */
export function calPoses(tx, ty, D, h) {
  const far = Math.min(FAR_CAP, D / Math.tan(DETECT_DEG * RAD)), out = [];
  for (const near of [true, false]) {
    for (let j = 0; j < N_CAL; j++) {
      const r = near ? NEAR[0] + (NEAR[1] - NEAR[0]) * j / (N_CAL - 1) : far * (FAR_FRAC[0] + (FAR_FRAC[1] - FAR_FRAC[0]) * j / (N_CAL - 1));
      const a = 2 * Math.PI * j / N_CAL;
      out.push([tx + r * Math.cos(a), ty + r * Math.sin(a), h]);
    }
  }
  return out;
}

/** fam (near then far) -> {f0, f1}. */
export function evidenceWindow(fam) {
  const f1 = median(fam.slice(0, N_CAL)), f0 = median(fam.slice(N_CAL));
  return { f0, f1: Math.max(f1, f0 + MIN_SPAN) };
}

export function viewAlt(r, d0, z0, d1, z1) {
  if (!(d1 > d0)) return z0;
  return z0 + (z1 - z0) * Math.min(Math.max((r - d0) / (d1 - d0), 0.0), 1.0);
}
