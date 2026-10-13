// Home view memory for the final approach (v2 brain), JS twin of scripts/brain/home.py.
// learn(look) stores a pad look one-shot; steer(looks, vis, bearC, rh, hv) scores every visible candidate, locks the
// most familiar one at >= fHome, and returns the approach vector (visual while the locked candidate is seen).
export const F_HOME = 0.97, F_TRACK = 0.485;
const wrap = (a) => { const r = (a + Math.PI) % (2 * Math.PI); return (r < 0 ? r + 2 * Math.PI : r) - Math.PI; };

export class HomeMemory {
  /** actFn(look) -> Uint8Array(nKc) active KCs (the mission's KC coding). */
  constructor(actFn, nKc, fHome = F_HOME, fTrack = F_TRACK) {
    this.actFn = actFn; this.fHome = fHome; this.fTrack = fTrack;
    this.w = new Uint8Array(nKc).fill(1);
    this.nViews = 0; this.locked = false; this.corr = [0, 0];
  }
  learn(look) {
    const a = this.actFn(look);
    for (let j = 0; j < a.length; j++) if (a[j]) this.w[j] = 0;
    this.nViews += 1;
  }
  familiarity(look) {
    const a = this.actFn(look);
    let n = 0, m = 0;
    for (let j = 0; j < a.length; j++) if (a[j]) { n++; if (this.w[j]) m++; }
    return 1.0 - m / n;
  }
  /** -> {corrB, fam, ap: [2], seen} (same order of operations as home.py). */
  steer(looks, vis, bearC, rh, hv) {
    const C = vis.length, fam = new Array(C).fill(0);
    for (let c = 0; c < C; c++) if (vis[c]) fam[c] = this.familiarity(looks[c]);
    let best = 0;
    for (let c = 1; c < C; c++) if ((vis[c] ? fam[c] : -1.0) > (vis[best] ? fam[best] : -1.0)) best = c;
    const fb = vis[best] ? fam[best] : 0.0;
    this.locked = this.locked || (vis[best] && fb >= this.fHome);
    const seen = this.locked && vis[best] && fb >= this.fTrack;
    const b = bearC[best], r = rh[best];
    if (seen) this.corr = [r * Math.cos(b) - hv[0], r * Math.sin(b) - hv[1]];
    const ap = this.locked ? [hv[0] + this.corr[0], hv[1] + this.corr[1]] : [hv[0], hv[1]];
    const corrB = this.locked ? wrap(Math.atan2(ap[1], ap[0]) - Math.atan2(hv[1], hv[0])) : 0.0;
    return { corrB, fam: fb, ap, seen };
  }
}
