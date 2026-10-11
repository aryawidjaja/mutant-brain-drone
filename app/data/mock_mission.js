// MOCK physics and brain, for building the UI only. NOT the 6-DOF quadrotor and NOT the insect brain.
// A kinematic point drone with a toy ring, toy path integrator and toy eye, emitting frames in the C6 shape
// so the app can be built before app/sim/mission.js exists. The worker labels it "MOCK physics".
//
// Conventions used by the UI (and assumed for the real modules): world ENU (x east, y north, z up), metres;
// bearings and yaw in the math sense (0 = east, counter-clockwise positive); pi.home_est / home_true are the
// drone-to-home VECTORS in world metres.
//
//   node app/data/mock_mission.js              self-check (asserts the C6 frame shape and a full mission)
//   node app/data/mock_mission.js --write-run  also writes app/runs/mock_run.json, mock_manifest.json, mock_bundle.js
import { Rng } from '../sim/rng.js';

const D2R = Math.PI / 180, R2D = 180 / Math.PI, DT = 0.02, N_RING = 16;
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const KIND = {
  tent: { base: [0.86, 0.47, 0.16], accent: [0.96, 0.92, 0.82], pattern: 'stripes' },
  pad: { base: [0.18, 0.45, 0.82], accent: [0.95, 0.95, 0.95], pattern: 'chevron' },
  backpack: { base: [0.66, 0.14, 0.2], accent: [0.15, 0.15, 0.18], pattern: 'dots' },
};
const mix3 = (a, b, s) => a.map((v, i) => v * s + b[i] * (1 - s));
const SIGMA = { none: 0, light: 0.5, moderate: 1.0, severe: 2.0 }; // mock gust std (m/s), not Dryden

export function buildWorld(sc) {
  const r = new Rng(sc.seed, 0), u = (a, b) => a + (b - a) * r.uniform();
  const [W, H] = sc.world.size_m, land = sc.world.landscape, dens = sc.world.obstacle_density;
  const home = sc.home, tp = sc.target.pos, cx = (home[0] + tp[0]) / 2, cy = (home[1] + tp[1]) / 2;
  const T = KIND[sc.target.kind];
  const objs = [{ role: 'target', kind: sc.target.kind, pos: tp, yaw: sc.target.yaw_deg || 0, s: 1 },
    ...(sc.decoys || []).map((d) => ({ role: 'decoy', kind: d.kind, pos: d.pos, yaw: u(0, 360), s: d.similarity ?? 0.8 }))];
  const order = objs.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const objects = objs.map((o, i) => {
    const K = KIND[o.kind], other = o.kind === sc.target.kind ? [0.45, 0.5, 0.4] : K.base;
    return { slot: order[i], role: o.role, kind: o.kind, x: o.pos[0], y: o.pos[1], z: o.pos[2] || 0, yaw_deg: o.yaw,
      appearance: { base: o.role === 'target' ? T.base : mix3(T.base, other, o.s), accent: o.s > 0.5 ? T.accent : K.accent,
        pattern: o.s > 0.5 ? T.pattern : K.pattern, similarity: o.s } };
  });
  const keep = [[home[0], home[1]], ...objects.map((o) => [o.x, o.y])];
  const clear = (x, y, rad) => keep.every(([a, b]) => Math.hypot(x - a, y - b) > rad + 3);
  const obstacles = [], area = W * H;
  const tree = (x, y) => ({ type: 'tree', x, y, r: u(0.8, 1.6), h: u(4, 9), rgb: [u(0.15, 0.25), u(0.38, 0.52), u(0.15, 0.22)] });
  const rock = (x, y) => ({ type: 'rock', x, y, r: u(0.4, 1.2), h: u(0.4, 1.0), rgb: [0.5, 0.49, 0.46].map((v) => v + u(-0.05, 0.05)) });
  if (land === 'urban') {
    for (let gx = -W / 2 + 6; gx < W / 2 - 6; gx += 14) for (let gy = -H / 2 + 6; gy < H / 2 - 6; gy += 14) {
      if (r.uniform() > dens) continue;
      const w = u(5, 9), d = u(5, 9), x = cx + gx, y = cy + gy;
      if (clear(x, y, Math.hypot(w, d) / 2)) obstacles.push({ type: 'building', x, y, w, d, h: u(4, 14), rgb: [0.6, 0.6, 0.62].map((v) => v + u(-0.1, 0.06)) });
    }
  } else {
    const n = Math.round(dens * area / (land === 'forest' ? 40 : 160));
    for (let k = 0; k < n; k++) {
      const x = cx + u(-W / 2, W / 2), y = cy + u(-H / 2, H / 2), o = land === 'forest' || r.uniform() < 0.5 ? tree(x, y) : rock(x, y);
      if (clear(x, y, o.r)) obstacles.push(o);
    }
  }
  const g = { meadow: [0.42, 0.55, 0.3], forest: [0.3, 0.4, 0.24], urban: [0.45, 0.45, 0.44] }[land];
  return { size: [W, H], ground: { rgb: g }, obstacles, objects, home: [...home],
    sun: { azimuth_deg: u(0, 360), elevation_deg: u(30, 60) }, light: sc.world.light ?? 1, fog_m: sc.world.fog_m ?? null, mock: true };
}

export function createMission(sc, worldIn = null) { // worldIn: a C5 world from app/sim/world.js, when it exists
  const world = worldIn || buildWorld(sc), objs = world.objects.slice().sort((a, b) => a.slot - b.slot), K = objs.length;
  const rw = new Rng(sc.seed, 1), rc = new Rng(sc.seed, 2), rf = new Rng(sc.seed, 5), rs = new Rng(sc.seed, 6), rv = new Rng(sc.seed, 4);
  const L = sc.brain.learning_flight_s ?? 10, arb = sc.brain.arbiter, gain = sc.brain.ring_gain ?? 1, tmax = sc.mission.max_time_s;
  const mean = sc.wind.mean_mps, sig = SIGMA[sc.wind.turbulence] || 0, faults = sc.faults || [];
  const vis = Math.min(20, world.fog_m || 1e9), light = world.light;
  const fault = (k) => faults.find((f) => f.kind === k);
  const on = (f, t) => f && t >= f.t_s && (f.kind !== 'occlusion' || t < f.t_s + (f.params.duration_s ?? 5));
  // presence: decoys appear at their t_s; the target is always there
  const scD = sc.decoys || [];
  const decoyOf = (o) => scD.find((d) => d.kind === o.kind && Math.hypot(d.pos[0] - o.x, d.pos[1] - o.y) < 0.5 && o.role === 'decoy');
  const present = (o, t) => {
    if (o.role === 'decoy') { const d = decoyOf(o); return !d || t >= (d.t_s || 0); }
    return true;
  };
  const EW = 36, EH = 9, dead = new Uint8Array(EW);
  { const f = fault('eye_columns'); if (f) for (let c = 0; c < EW; c++) dead[c] = rf.uniform() < (f.params.frac ?? 0.25) ? 1 : 0; }
  const S = {
    t: 0, phase: L > 0 ? 'learning' : 'search', pos: [sc.home[0], sc.home[1], 0.05], vel: [0, 0, 0], yaw: 0, roll: 0, pitch: 0, omega: [0, 0, 0],
    gust: [0, 0, 0], ring: new Float32Array(N_RING), drives: { objects: new Float32Array(N_RING), home: new Float32Array(N_RING), search: new Float32Array(N_RING) },
    pi: [0, 0], acc: new Float32Array(K), sm: 'SEARCH', smT: 0, commit: 0, cmd: 0, path: 0, bank: 0, energy: 0, collisions: 0,
    seen: objs.map(() => ({ bearing: 0, strength: 0, visible: false })), fam: 0, avoid: { yaw_offset_deg: 0, loom: 0 },
    stopAt: null, wrong: false, outcome: null, hdgEst: 0,
  };
  const bump = (arr, b, a) => { for (let i = 0; i < N_RING; i++) arr[i] += a * Math.exp(3 * (Math.cos(2 * Math.PI * i / N_RING - b) - 1)); };
  const popvec = (arr) => { let x = 0, y = 0; for (let i = 0; i < N_RING; i++) { x += arr[i] * Math.cos(2 * Math.PI * i / N_RING); y += arr[i] * Math.sin(2 * Math.PI * i / N_RING); } return Math.atan2(y, x); };

  function senseObjects() {
    const occ = on(fault('occlusion'), S.t), eyeF = fault('eye_columns') && S.t >= fault('eye_columns').t_s ? 1 - (fault('eye_columns').params.frac ?? 0.25) : 1;
    objs.forEach((o, k) => {
      const dx = o.x - S.pos[0], dy = o.y - S.pos[1], d = Math.hypot(dx, dy);
      const v = present(o, S.t) && d < vis && !(occ && o.role === 'target') && S.phase !== 'learning';
      S.seen[k] = { bearing: Math.atan2(dy, dx) + S.hdgEst - S.yaw, visible: v,
        strength: v ? clamp(o.appearance.similarity * (1.15 - d / vis) * eyeF + 0.05 * rv.normal(), 0, 1) : 0 };
    });
    S.fam = Math.max(0.08, ...S.seen.map((s) => s.strength));
  }
  function arbitrate() {
    const D = S.drives; D.objects.fill(0); D.home.fill(0); D.search.fill(0);
    const homeB = Math.atan2(-S.pi[1], -S.pi[0]);
    const hunting = S.phase !== 'home'; // after a stop the objects no longer pull
    S.seen.forEach((s) => { if (s.visible && hunting) bump(D.objects, s.bearing, s.strength); });
    if (S.phase === 'home') bump(D.home, homeB, 1);
    const tb = Math.atan2(sc.target.pos[1] - sc.home[1], sc.target.pos[0] - sc.home[0]); // mock cheat: search drifts toward the target area
    if (S.phase === 'search') bump(D.search, tb + 1.1 * Math.sin(0.13 * S.t) + 0.3 * rs.normal(), 0.5);
    const tot = new Float32Array(N_RING); for (let i = 0; i < N_RING; i++) tot[i] = D.objects[i] + D.home[i] + D.search[i];
    let hc, viz = {};
    if (arb === 'ring' || arb === 'ring_static') {
      const g = arb === 'ring' ? gain : 0, r = S.ring, nr = new Float32Array(N_RING);
      for (let i = 0; i < N_RING; i++) { let rec = 0; for (let j = 0; j < N_RING; j++) rec += r[j] * Math.cos(2 * Math.PI * (i - j) / N_RING); nr[i] = r[i] + DT / 0.3 * (-r[i] + Math.max(0, tot[i] + 0.25 * g * rec / N_RING * 4)); }
      S.ring.set(nr); hc = popvec(nr); viz = { ring: S.ring, drives: D };
    } else if (arb === 'vote') { let m = 0; for (let i = 1; i < N_RING; i++) if (tot[i] > tot[m]) m = i; hc = 2 * Math.PI * m / N_RING; viz = { winner: m }; }
    else if (arb === 'sum') hc = popvec(tot);
    else if (arb === 'accumulator') {
      S.seen.forEach((s, k) => { S.acc[k] += DT / 2 * (-S.acc[k] + s.strength); });
      let m = 0; for (let k = 1; k < K; k++) if (S.acc[k] > S.acc[m]) m = k;
      hc = S.acc[m] > 0.2 && hunting ? S.seen[m].bearing : popvec(tot); viz = { acc: Array.from(S.acc) };
    } else { // state_machine
      S.smT += DT; const best = S.seen.reduce((b, s, k) => (s.strength > (b < 0 ? 0 : S.seen[b].strength) ? k : b), -1);
      if (S.phase === 'home') S.sm = 'HOME'; else if (S.sm === 'SEARCH' && best >= 0 && S.seen[best].strength > 0.4) { S.sm = 'APPROACH'; S.smT = 0; }
      else if (S.sm === 'APPROACH' && (best < 0 || S.seen[best].strength < 0.2) && S.smT > 3) S.sm = 'SEARCH';
      hc = S.sm === 'APPROACH' && best >= 0 ? S.seen[best].bearing : popvec(tot); viz = { state: S.sm, timer_s: +S.smT.toFixed(1) };
    }
    S.commit = clamp(Math.max(...tot) / 1.2, 0, 1); S.cmd = hc;
    return { hc, viz };
  }
  let lastViz = {};
  function step() {
    if (S.outcome) return;
    const t = S.t, dt = DT;
    // heading estimate: compass faults (half-silenced = noisy, false sun = bias)
    const fs = fault('false_sun'), ch = fault('compass_half');
    S.hdgEst = S.yaw + (fs && t >= fs.t_s ? (fs.params.offset_deg ?? 60) * D2R : 0) + (ch && t >= ch.t_s ? 0.35 * rc.normal() : 0.02 * rc.normal());
    senseObjects();
    const { hc, viz } = arbitrate(); lastViz = viz;
    // phase logic (mock stop rule: within 2 m of an object with strength > 0.6)
    if (S.phase === 'learning' && t >= L) S.phase = 'search';
    if (S.phase === 'search' && S.seen.some((s) => s.strength > 0.45)) S.phase = 'approach';
    if (S.phase === 'approach') {
      if (S.seen.every((s) => s.strength < 0.15)) S.phase = 'search';
      objs.forEach((o, k) => { if (S.phase === 'approach' && Math.hypot(o.x - S.pos[0], o.y - S.pos[1]) < 2 && S.seen[k].strength > 0.6) { S.stopAt = o.role; S.wrong = o.role === 'decoy'; S.phase = 'home'; } });
    }
    let want;
    if (S.phase === 'learning') want = Math.atan2(S.pos[1] - sc.home[1], S.pos[0] - sc.home[0]) + Math.PI / 2 + (Math.hypot(S.pos[0] - sc.home[0], S.pos[1] - sc.home[1]) < 2 ? -0.6 : 0.2);
    else want = hc - (S.hdgEst - S.yaw);
    // avoidance (geometric look-ahead, mock)
    let off = 0, loom = 0;
    for (const o of world.obstacles) {
      const rad = o.type === 'building' ? Math.hypot(o.w, o.d) / 2 : o.r; if (o.h < S.pos[2]) continue;
      const dx = o.x - S.pos[0], dy = o.y - S.pos[1], d = Math.hypot(dx, dy) - rad, rel = wrap(Math.atan2(dy, dx) - S.yaw);
      if (d < 4 && Math.abs(rel) < 1.1) { const k = (1 - Math.max(0, d) / 4); if (k > loom) { loom = k; off = -Math.sign(rel || 1) * k * 1.2; } }
      if (d < 0.05) { S.collisions += S.colFlag ? 0 : 1; S.colFlag = 2; S.pos[0] -= dx / (d + rad) * 0.1; S.pos[1] -= dy / (d + rad) * 0.1; }
    }
    if (S.colFlag) S.colFlag--;
    S.avoid = { yaw_offset_deg: off * R2D, loom };
    const dyaw = clamp(wrap(want + off - S.yaw), -2.5 * dt, 2.5 * dt); S.yaw = wrap(S.yaw + dyaw);
    // wind: mean + first-order gust (mock, not Dryden)
    for (let i = 0; i < 3; i++) S.gust[i] += -S.gust[i] * dt + sig * Math.sqrt(2 * dt) * rw.normal() * (i === 2 ? 0.4 : 1);
    const wind = mean.map((m, i) => m + S.gust[i]);
    const homeD = Math.hypot(sc.home[0] - S.pos[0], sc.home[1] - S.pos[1]);
    const spd = S.phase === 'learning' ? 1 : S.phase === 'approach' ? 1.4 : S.phase === 'home' ? Math.min(2.2, 0.6 + homeD) : 2.2;
    const alt = S.phase === 'learning' ? 1.2 + 0.1 * t : S.phase === 'done' ? 0.05 : 2.5;
    const prev = S.vel.slice(), vc = [spd * Math.cos(S.yaw) + 0.25 * wind[0], spd * Math.sin(S.yaw) + 0.25 * wind[1], (Math.min(alt, 2.5) - S.pos[2]) * 1.5];
    for (let i = 0; i < 3; i++) { S.vel[i] += (vc[i] - S.vel[i]) * dt / 0.4; S.pos[i] += S.vel[i] * dt; }
    const step = Math.hypot(S.vel[0], S.vel[1]) * dt; S.path += step;
    // path integration with odometry scale fault
    const sp = fault('speed_scale'), scl = sp && t >= sp.t_s ? (sp.params.scale ?? 1.2) : 1, e = S.hdgEst - S.yaw;
    const vx = S.vel[0] * Math.cos(e) - S.vel[1] * Math.sin(e), vy = S.vel[0] * Math.sin(e) + S.vel[1] * Math.cos(e);
    S.pi[0] += vx * scl * dt; S.pi[1] += vy * scl * dt;
    // attitude from the acceleration needed (mock tilt)
    const ax = (S.vel[0] - prev[0]) / dt + 0.35 * (S.vel[0] - wind[0]), ay = (S.vel[1] - prev[1]) / dt + 0.35 * (S.vel[1] - wind[1]);
    const af = ax * Math.cos(S.yaw) + ay * Math.sin(S.yaw), al = -ax * Math.sin(S.yaw) + ay * Math.cos(S.yaw);
    const np = Math.atan(af / 9.81), nr = -Math.atan(al / 9.81);
    S.omega = [(nr - S.roll) / dt, (np - S.pitch) / dt, dyaw / dt]; S.roll = nr; S.pitch = np;
    S.bank = Math.max(S.bank, Math.abs(nr) * R2D, Math.abs(np) * R2D); S.energy += dt * (1 + Math.hypot(af, al) / 9.81);
    S.t = +(t + dt).toFixed(6);
    if (S.phase === 'home' && Math.hypot(S.pi[0], S.pi[1]) < 1) S.phase = 'done';
    if (S.phase === 'done' || S.t >= tmax) {
      S.phase = 'done';
      S.outcome = { success_target: S.stopAt === 'target', success_home: S.stopAt != null && homeD < 2, wrong_object_stop: S.wrong, collisions: S.collisions,
        time_s: S.t, path_len_m: +S.path.toFixed(2), home_error_m: +homeD.toFixed(2), max_bank_deg: +S.bank.toFixed(1), energy_proxy: +S.energy.toFixed(1),
        t_hold_occlusion_s: null, t_release_decoy_s: null, mock: true };
    }
  }
  function eye() {
    const rgb = new Uint8Array(EW * EH * 3), zb = new Float32Array(EW * EH).fill(1e9), ra = new Rng(1, 4);
    const sky = [0.62, 0.74, 0.9], gr = world.ground.rgb, put = (c, r, col, d) => {
      const i = r * EW + c; if (d >= zb[i]) return; zb[i] = d;
      const f = world.fog_m ? Math.exp(-d / world.fog_m) : 1;
      for (let k = 0; k < 3; k++) rgb[3 * i + k] = clamp(255 * light * (col[k] * f + sky[k] * (1 - f)), 0, 255);
    };
    for (let c = 0; c < EW; c++) for (let r = 0; r < EH; r++) put(c, r, r < 4 ? sky : gr, 1e8 + r);
    const draw = (x, y, rad, h, col, z0 = 0) => {
      const dx = x - S.pos[0], dy = y - S.pos[1], d = Math.hypot(dx, dy); if (d < 0.3 || d > 40) return;
      const b = wrap(Math.atan2(dy, dx) - S.yaw), hw = Math.atan(rad / d), top = Math.atan2(h - S.pos[2], d), bot = Math.atan2(z0 - S.pos[2], d);
      for (let c = 0; c < EW; c++) {
        const az = Math.PI - 2 * Math.PI * (c + 0.5) / EW; if (Math.abs(wrap(az - b)) > hw + Math.PI / EW / 2) continue;
        for (let r = 0; r < EH; r++) { const el = (0.5 - (r + 0.5) / EH) * Math.PI / 2; if (el <= top && el >= bot) put(c, r, col, d); }
      }
    };
    for (const o of world.obstacles) draw(o.x, o.y, o.type === 'building' ? Math.max(o.w, o.d) / 2 : o.r, o.h, o.rgb);
    objs.forEach((o) => { if (present(o, S.t) && !(on(fault('occlusion'), S.t) && o.role === 'target')) draw(o.x, o.y, 1.2, o.kind === 'pad' ? 0.05 : 1.2, o.appearance.base); });
    const ef = fault('eye_columns');
    for (let c = 0; c < EW; c++) for (let r = 0; r < EH; r++) {
      const i = 3 * (r * EW + c);
      if (ef && S.t >= ef.t_s && dead[c]) rgb[i] = rgb[i + 1] = rgb[i + 2] = 0;
      else for (let k = 0; k < 3; k++) rgb[i + k] = clamp(rgb[i + k] + 6 * ra.normal(), 0, 255);
    }
    return { w: EW, h: EH, rgb };
  }
  const relDir = (vx, vy) => { const a = wrap(Math.atan2(vy, vx) - S.yaw) * R2D; return Math.abs(a) < 30 ? 'ahead' : Math.abs(a) > 150 ? 'behind me' : a > 0 ? (Math.abs(a) > 90 ? 'behind me, left' : 'to my left') : (Math.abs(a) > 90 ? 'behind me, right' : 'to my right'); };
  function frame() {
    const q = quat(S.roll, S.pitch, S.yaw), hv = [-S.pi[0], -S.pi[1]], ht = [sc.home[0] - S.pos[0], sc.home[1] - S.pos[1]];
    const tgt = objs.find((o) => o.role === 'target'), near = objs.reduce((b, o) => (Math.hypot(o.x - S.pos[0], o.y - S.pos[1]) < Math.hypot(b.x - S.pos[0], b.y - S.pos[1]) ? o : b), objs[0]);
    const hover = 0.52, mr = S.omega[0] * 0.01, mp = S.omega[1] * 0.01, my = S.omega[2] * 0.02;
    return {
      t: S.t, phase: S.phase, outcome: S.outcome,
      quad: { pos: S.pos.slice(), vel: S.vel.slice(), quat: q, euler_deg: [S.roll * R2D, S.pitch * R2D, S.yaw * R2D], omega: S.omega.slice(),
        motors: [hover - mr - mp + my, hover - mr + mp - my, hover + mr + mp + my, hover + mr - mp - my].map((m) => clamp(S.phase === 'done' ? 0 : m, 0, 1)) },
      wind: mean.map((m, i) => m + S.gust[i]), eye: eye(), familiarity: S.fam,
      objects: S.seen.map((s, k) => ({ slot: k, bearing_deg: s.bearing * R2D, strength: s.strength, visible: s.visible })),
      ring: { n: N_RING, activity: Float32Array.from(S.ring), drives: { objects: Float32Array.from(S.drives.objects), home: Float32Array.from(S.drives.home), search: Float32Array.from(S.drives.search) }, heading_cmd_deg: S.cmd * R2D },
      arbiter: { name: arb, commit: S.commit, viz: Object.fromEntries(Object.entries(lastViz).filter(([k]) => k !== 'ring' && k !== 'drives')) },
      pi: { home_est: hv, home_true: ht },
      avoid: { ...S.avoid },
      mind: { text: `home is ${Math.hypot(...hv).toFixed(0)} m ${relDir(...hv)}; target familiarity ${S.fam.toFixed(2)}`,
        truth: `home is ${Math.hypot(...ht).toFixed(0)} m ${relDir(...ht)}; nearest object: ${near.role} at ${Math.hypot(near.x - S.pos[0], near.y - S.pos[1]).toFixed(0)} m; target ${Math.hypot(tgt.x - S.pos[0], tgt.y - S.pos[1]).toFixed(0)} m` },
      faults_active: faults.filter((f) => on(f, S.t)).map((f) => f.kind),
    };
  }
  return { step, frame, world, mock: true };
}

function quat(r, p, y) { // ZYX (yaw, pitch, roll) -> [w, x, y, z]
  const cr = Math.cos(r / 2), sr = Math.sin(r / 2), cp = Math.cos(p / 2), sp = Math.sin(p / 2), cy = Math.cos(y / 2), sy = Math.sin(y / 2);
  return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy];
}

export const DEFAULT_SCENARIO = {
  seed: 0, world: { size_m: [80, 80], landscape: 'meadow', obstacle_density: 0.3, light: 1, fog_m: null },
  home: [0, 0, 0], target: { kind: 'tent', pos: [30, 20, 0], yaw_deg: 0 }, decoys: [],
  wind: { mean_mps: [0.5, 0, 0], turbulence: 'light' }, faults: [],
  brain: { arbiter: 'ring', ring_gain: 1, learning_flight_s: 10 }, mission: { max_time_s: 300, dt_s: 0.002 },
};

// ---------------------------------------------------------------- self-check (node only)
if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  const assert = (c, m) => { if (!c) throw new Error('self-check failed: ' + m); };
  const sc = structuredClone(DEFAULT_SCENARIO);
  sc.decoys = [{ kind: 'tent', pos: [24, 30, 0], similarity: 0.7, t_s: 0 }];
  sc.faults = [{ kind: 'occlusion', t_s: 20, params: { duration_s: 3 } }];
  const m = createMission(sc), phases = new Set(), rec = [];
  const w1 = JSON.stringify(buildWorld(sc)); assert(w1 === JSON.stringify(buildWorld(sc)), 'world deterministic');
  assert(m.world.objects.length === 2 && m.world.objects.some((o) => o.role === 'target'), 'objects');
  let f;
  for (let i = 0; i < 300 / DT; i++) {
    m.step(); f = m.frame(); phases.add(f.phase);
    if (i % 5 === 0) rec.push(f);
    if (f.outcome) break;
  }
  for (const k of ['t', 'phase', 'outcome', 'quad', 'wind', 'eye', 'familiarity', 'objects', 'ring', 'arbiter', 'pi', 'avoid', 'mind', 'faults_active']) assert(k in f, 'frame key ' + k);
  assert(f.eye.rgb.length === f.eye.w * f.eye.h * 3, 'eye size');
  assert(f.ring.activity.length === f.ring.n, 'ring n');
  assert(f.quad.quat.length === 4 && Math.abs(Math.hypot(...f.quad.quat) - 1) < 1e-9, 'unit quat');
  assert(f.outcome && phases.has('learning') && phases.has('search') && phases.has('home'), 'mission completes: ' + [...phases]);
  console.log('mock_mission self-check OK:', JSON.stringify(f.outcome));
  if (process.argv.includes('--write-run')) {
    const fs = await import('node:fs'), path = await import('node:path');
    const dir = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'runs');
    const b64 = (u8) => Buffer.from(u8).toString('base64');
    const r3 = (a) => Array.from(a, (v) => Math.round(v * 1000) / 1000);
    const frames = rec.map((x, i) => ({
      ...x, quad: { ...x.quad, pos: r3(x.quad.pos), vel: r3(x.quad.vel), quat: r3(x.quad.quat), euler_deg: r3(x.quad.euler_deg), omega: r3(x.quad.omega), motors: r3(x.quad.motors) },
      wind: r3(x.wind), eye: i % 5 === 0 ? { w: x.eye.w, h: x.eye.h, b64: b64(x.eye.rgb) } : undefined,
      ring: { ...x.ring, activity: r3(x.ring.activity), drives: { objects: r3(x.ring.drives.objects), home: r3(x.ring.drives.home), search: r3(x.ring.drives.search) } },
    }));
    const run = { meta: { mock: true, label: 'MOCK run: ring, seed 0, one decoy, 3 s occlusion', arbiter: 'ring', seed: 0, git: 'mock', sim_version: 'mock', dt_frames_s: 0.1, scenario: sc }, world: m.world, frames };
    const man = { mock: true, runs: [{ file: 'mock_run.json', label: run.meta.label }] };
    fs.writeFileSync(path.join(dir, 'mock_run.json'), JSON.stringify(run));
    fs.writeFileSync(path.join(dir, 'mock_manifest.json'), JSON.stringify(man, null, 1));
    fs.writeFileSync(path.join(dir, 'mock_bundle.js'), '// MOCK replay bundle for file:// (generated by app/data/mock_mission.js --write-run)\nwindow.MOCK_RUN_BUNDLE = ' + JSON.stringify({ manifest: man, runs: { 'mock_run.json': run } }) + ';\n');
    console.log('wrote', frames.length, 'frames to', dir);
  }
}
