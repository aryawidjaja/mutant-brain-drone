// Mutant brain drone: 2D panels (canvas) and Results charts (inline SVG). Classic script; defines window.Panels.
// Every drawer takes plain data (a C6 frame, a C5 world, a summary object) plus the current CSS tokens, so the same
// code draws the live simulator and the recorded replays. World angles: 0 = east, counter-clockwise positive; on
// screen north is up and east is right.
'use strict';
(function () {
const TAU = Math.PI * 2, D2R = Math.PI / 180;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ROLE = { ring: 'a', ring_static: 'n', vote: 'b', sum: 'b', accumulator: 'b', state_machine: 'b' };
const roleCol = (css, arb) => ({ a: css.a, b: css.b, n: css.muted }[ROLE[arb] || 'n']);
function ctx2d(c) {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = c.clientWidth || c.width, H = c.clientHeight || c.height;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H); return { x, W, H };
}
const rgbCss = (a, k = 1) => `rgb(${a.map((v) => Math.round(clamp(v * k, 0, 1) * 255)).join(',')})`;
function eyeBytes(eye) {
  if (!eye) return null;
  if (eye.rgb) return eye.rgb instanceof Uint8Array ? eye.rgb : Uint8Array.from(eye.rgb);
  if (eye.b64) { const s = atob(eye.b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); eye.rgb = u; return u; }
  return null;
}
function arrow(x, x0, y0, x1, y1, col, w = 2, head = 7, dash) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  x.strokeStyle = x.fillStyle = col; x.lineWidth = w; x.setLineDash(dash || []);
  x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1 - Math.cos(a) * head * 0.6, y1 - Math.sin(a) * head * 0.6); x.stroke(); x.setLineDash([]);
  if (Math.hypot(x1 - x0, y1 - y0) < 2) return;
  x.beginPath(); x.moveTo(x1, y1); x.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4)); x.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4)); x.fill();
}

// ---------- eye: native-resolution image, upscaled by CSS with crisp pixels
function eye(c, e) {
  const u = eyeBytes(e); if (!u) return;
  if (c.width !== e.w || c.height !== e.h) { c.width = e.w; c.height = e.h; c.style.aspectRatio = `${e.w} / ${e.h}`; }
  const img = new ImageData(e.w, e.h);
  for (let i = 0, j = 0; i < e.w * e.h; i++) { img.data[j++] = u[3 * i]; img.data[j++] = u[3 * i + 1]; img.data[j++] = u[3 * i + 2]; img.data[j++] = 255; }
  c.getContext('2d').putImageData(img, 0, 0);
}

// ---------- target picture: drawn from the C5 appearance (or a gallery image when one is given)
function pic(c, obj, img) {
  const x = c.getContext('2d'), W = c.width, H = c.height;
  x.fillStyle = '#6f8a52'; x.fillRect(0, 0, W, H);
  if (img && img.complete && img.naturalWidth) { x.imageSmoothingEnabled = true; x.drawImage(img, 0, 0, W, H); return; }
  if (!obj) return;
  const ap = { pattern: { tent: 'stripes', pad: 'chevron', backpack: 'dots' }[obj.kind], ...(obj.appearance || {}) }, base = rgbCss(ap.base || [0.8, 0.5, 0.2]), acc = rgbCss(ap.accent || [1, 1, 1]);
  x.save(); x.translate(W / 2, H / 2);
  const shape = () => {
    x.beginPath();
    if (obj.kind === 'tent') { x.moveTo(-24, 18); x.lineTo(0, -20); x.lineTo(24, 18); x.closePath(); }
    else if (obj.kind === 'pad') { x.rect(-22, -22, 44, 44); }
    else { x.roundRect ? x.roundRect(-14, -20, 28, 38, 7) : x.rect(-14, -20, 28, 38); }
  };
  shape(); x.fillStyle = base; x.fill(); x.save(); shape(); x.clip(); x.fillStyle = x.strokeStyle = acc; x.lineWidth = 3;
  if (ap.pattern === 'stripes') for (let k = -30; k < 30; k += 9) { x.beginPath(); x.moveTo(k, -30); x.lineTo(k + 12, 30); x.stroke(); }
  else if (ap.pattern === 'chevron') for (let k = -16; k < 24; k += 12) { x.beginPath(); x.moveTo(-18, k + 8); x.lineTo(0, k - 4); x.lineTo(18, k + 8); x.stroke(); }
  else for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) { x.beginPath(); x.arc(i * 8, j * 8, 2, 0, TAU); x.fill(); }
  x.restore(); x.restore();
}

// ---------- familiarity + commit, trailing 30 s window
function fam(c, h, T, css) {
  const { x, W, H } = ctx2d(c), t1 = Math.max(T, 30), t0 = t1 - 30, X = (t) => ((t - t0) / 30) * W, Y = (v) => H - 3 - clamp(v, 0, 1) * (H - 8);
  x.strokeStyle = css.line; x.lineWidth = 1; x.beginPath();
  for (const v of [0, 0.5, 1]) { x.moveTo(0, Math.round(Y(v)) + 0.5); x.lineTo(W, Math.round(Y(v)) + 0.5); } x.stroke();
  x.fillStyle = css.muted; x.font = `10px ${css.mono}`; x.fillText('1', 2, Y(1) + 10); x.fillText('0', 2, Y(0) - 2);
  const line = (arr, col, w) => {
    x.strokeStyle = col; x.lineWidth = w; x.beginPath(); let on = false;
    for (let i = 0; i < h.t.length; i++) { const t = h.t[i]; if (t < t0 || t > T) continue; const px = X(t), py = Y(arr[i]); on ? x.lineTo(px, py) : x.moveTo(px, py); on = true; }
    x.stroke();
  };
  line(h.commit, css.muted, 1); line(h.fam, css.ink, 1.6);
}

// ---------- steering hub, drawn in the drone's compass frame (north up = the drone's own sense of north; it differs
// from true north under compass faults). Ring arms and the vote: ring of heading neurons (neuron i prefers 2*pi*i/n)
// with the pull of each drive outside it (the vote shows its input bump; it has no dynamics). Vector sum: the drive
// vectors. Accumulator and state machine: per-slot evidence at each slot's perceived bearing.
const SM_STATES = ['SEARCH', 'APPROACH', 'HOLD', 'HOME'];
function hub(c, f, css) {
  const { x, W, H } = ctx2d(c), cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.3, arb = f.arbiter ? f.arbiter.name : '';
  const viz = (f.arbiter && f.arbiter.viz) || {};
  const P = (ang, r) => [cx + r * Math.cos(ang), cy - r * Math.sin(ang)];
  const role = roleCol(css, arb), own = f.pi && Number.isFinite(f.pi.heading_est_deg) ? f.pi.heading_est_deg * D2R : f.quad.euler_deg[2] * D2R;
  const hv = f.pi ? f.pi.home_est_compass || f.pi.home_est : null;
  x.strokeStyle = css.line; x.lineWidth = 1; x.beginPath(); x.arc(cx, cy, R, 0, TAU); x.stroke();
  x.fillStyle = css.muted; x.font = `10px ${css.mono}`; x.textAlign = 'center';
  x.fillText('N', cx, cy - R * 1.62); x.fillText('E', cx + R * 1.62, cy + 3); x.fillText('S', cx, cy + R * 1.62 + 6); x.fillText('W', cx - R * 1.62, cy + 3);
  const bar = (a, v, col, solid = true, w = 3) => {
    const [x0, y0] = P(a, R * 1.06), [x1, y1] = P(a, R * (1.06 + 0.5 * clamp(v, 0, 1)));
    x.strokeStyle = col; x.lineWidth = w; x.setLineDash(solid ? [] : [3, 3]); x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1, y1); x.stroke(); x.setLineDash([]);
  };
  const label = (a, r, s) => { const [lx, ly] = P(a, r); x.fillStyle = css.muted; x.textAlign = 'center'; x.fillText(s, lx, ly + 3); };
  const ring = f.ring && f.ring.activity && f.ring.activity.length ? f.ring : null;
  if (ring && (/^ring/.test(arb) || arb === 'vote')) {
    const n = ring.n || ring.activity.length, act = ring.activity, dr = ring.drives || {};
    let amax = 1e-6; for (let i = 0; i < n; i++) amax = Math.max(amax, act[i]);
    let dmax = 1e-6; for (const k of ['objects', 'home', 'search']) if (dr[k]) for (let i = 0; i < n; i++) dmax = Math.max(dmax, dr[k][i]);
    const cols = { objects: css.dobj, home: css.dhome, search: css.dsearch }, offs = { objects: -0.22, home: 0, search: 0.22 };
    for (let i = 0; i < n; i++) {
      const ang = (TAU * i) / n;
      for (const k of ['objects', 'home', 'search']) {
        const v = dr[k] ? dr[k][i] / dmax : 0; if (v < 0.02) continue;
        const a2 = ang + offs[k] * TAU / n, [x0, y0] = P(a2, R * 1.12), [x1, y1] = P(a2, R * (1.12 + 0.42 * v));
        x.strokeStyle = cols[k]; x.lineWidth = 3; x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1, y1); x.stroke();
      }
      const [px, py] = P(ang, R), a = clamp(act[i] / amax, 0, 1), rr = Math.max(3, Math.min(9, (TAU * R) / n * 0.32));
      x.globalAlpha = 1; x.fillStyle = css.panel2; x.beginPath(); x.arc(px, py, rr, 0, TAU); x.fill();
      x.globalAlpha = 0.12 + 0.88 * a; x.fillStyle = role; x.beginPath(); x.arc(px, py, rr, 0, TAU); x.fill();
      x.globalAlpha = 1; x.strokeStyle = css.line2; x.lineWidth = 1; x.stroke();
    }
    x.fillStyle = css.muted; x.textAlign = 'left'; x.fillText(`${arb === 'vote' ? 'input peak' : 'peak'} ${amax.toFixed(2)}`, 4, H - 4);
  } else if (arb === 'sum' && viz.vectors) {
    const V = viz.vectors, cols = { objects: css.dobj, home: css.dhome, search: css.dsearch };
    const tot = [0, 1].map((i) => Object.values(V).reduce((s, v) => s + v[i], 0));
    const m = Math.max(1e-6, Math.hypot(...tot), ...Object.values(V).map((v) => Math.hypot(...v))), s = (R * 1.5) / m;
    for (const [k, v] of Object.entries(V)) if (Math.hypot(...v) * s > 2) arrow(x, cx, cy, cx + v[0] * s, cy - v[1] * s, cols[k] || css.muted, 2.5, 7);
    x.fillStyle = css.muted; x.textAlign = 'left'; x.fillText(`|sum| ${Math.hypot(...tot).toFixed(2)}`, 4, H - 4);
  } else {
    const ev = viz.evidence ? Array.from(viz.evidence) : null, objs = f.objects || [];
    let emax = 1e-6; if (ev) for (const v of ev) emax = Math.max(emax, v);
    const sm = arb === 'state_machine', acc = arb === 'accumulator';
    for (const o of objs) {
      const a = o.bearing_deg * D2R, v = ev ? (acc ? ev[o.slot] / emax : ev[o.slot]) : o.strength;
      const chosen = sm ? viz.target === o.slot : acc ? viz.choice === o.slot : false;
      bar(a, Math.max(v, 0.06), chosen ? role : css.dobj, o.visible, chosen ? 4.5 : 3);
      label(a, R * 1.06 + R * 0.5 * clamp(Math.max(v, 0.06), 0, 1) + 9, String(o.slot) + (sm && viz.rejected && viz.rejected[o.slot] ? '×' : ''));
    }
    if (hv) {
      const ha = Math.atan2(hv[1], hv[0]), K = objs.length;
      if (acc && ev && ev.length > K) bar(ha, ev[K] / emax, viz.choice === K ? role : css.dhome);
      else if (f.phase === 'home') bar(ha, 0.9, css.dhome);
    }
    x.textAlign = 'center';
    if (sm) { x.fillStyle = css.ink; x.font = `600 11px ${css.mono}`; x.fillText(SM_STATES[viz.state] || String(viz.state ?? ''), cx, cy + R + 4 - R * 0.35); x.font = `10px ${css.mono}`; }
    if (acc && ev && ev.length > objs.length + 1) { x.fillStyle = css.muted; x.textAlign = 'left'; x.fillText(`search ${ev[objs.length + 1].toFixed(2)}${viz.choice === objs.length + 1 ? ' ◂' : ''}`, 4, H - 4); }
  }
  // heading command (role colour) and the drone's own heading estimate (ink)
  const hc = f.ring && Number.isFinite(f.ring.heading_cmd_deg) ? f.ring.heading_cmd_deg * D2R : null;
  if (hc !== null) { const [hx, hy] = P(hc, R * 0.82); arrow(x, cx, cy, hx, hy, role, 2.5, 8); }
  const [yx, yy] = P(own, R * 0.6); arrow(x, cx, cy, yx, yy, css.ink, 1.2, 6);
  const commit = f.arbiter && Number.isFinite(f.arbiter.commit) ? f.arbiter.commit : null;
  if (commit !== null) { x.fillStyle = css.muted; x.textAlign = 'right'; x.fillText(`commit ${commit.toFixed(2)}`, W - 4, H - 4); }
  x.textAlign = 'left';
}
// ---------- home vector: estimate vs truth, drone at the centre, north up
function homev(c, pi, css) {
  const { x, W, H } = ctx2d(c), cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.42;
  if (!pi) return;
  const e = pi.home_est, t = pi.home_true, m = Math.max(5, Math.hypot(...e), Math.hypot(...t));
  const nice = [5, 10, 20, 25, 50, 100, 200, 500, 1000].find((v) => v >= m) || m, s = R / nice;
  x.strokeStyle = css.line; x.lineWidth = 1;
  for (const k of [0.5, 1]) { x.beginPath(); x.arc(cx, cy, R * k, 0, TAU); x.stroke(); }
  x.beginPath(); x.moveTo(cx - R, cy); x.lineTo(cx + R, cy); x.moveTo(cx, cy - R); x.lineTo(cx, cy + R); x.stroke();
  x.fillStyle = css.muted; x.font = `10px ${css.mono}`; x.fillText(`${nice} m`, cx + 3, cy - R + 10); x.fillText('N', cx + 3, cy - R - 2 + 0);
  arrow(x, cx, cy, cx + t[0] * s, cy - t[1] * s, css.muted, 1.6, 7, [4, 3]);
  arrow(x, cx, cy, cx + e[0] * s, cy - e[1] * s, css.ink, 2.2, 8);
  x.fillStyle = css.ink; x.beginPath(); x.arc(cx, cy, 2.5, 0, TAU); x.fill();
}

// ---------- attitude (horizon), motors, wind
function att(c, f, css) {
  const { x, W, H } = ctx2d(c), q = f.quad, [roll, pitch, yaw] = q.euler_deg, r = Math.min(56, H / 2 - 8), cx = r + 6, cy = H / 2;
  // horizon: pitch > 0 = nose down (rotation about +y, body FLU), so the horizon rises
  x.save(); x.beginPath(); x.arc(cx, cy, r, 0, TAU); x.clip();
  x.translate(cx, cy); x.rotate(roll * D2R); const off = clamp(pitch, -45, 45) / 45 * r;
  const lt = document.documentElement.dataset.theme === 'light';
  x.fillStyle = lt ? '#a9c6e8' : '#2a4a6b'; x.fillRect(-2 * r, -2 * r + off, 4 * r, 2 * r);
  x.fillStyle = lt ? '#b59c78' : '#5a4632'; x.fillRect(-2 * r, off, 4 * r, 2 * r);
  x.strokeStyle = '#fff'; x.lineWidth = 1; for (const p of [-20, -10, 10, 20]) { const y = off - p / 45 * r; x.beginPath(); x.moveTo(-r * (Math.abs(p) === 10 ? 0.2 : 0.35), y); x.lineTo(r * (Math.abs(p) === 10 ? 0.2 : 0.35), y); x.stroke(); }
  x.restore();
  x.strokeStyle = css.line2; x.lineWidth = 1; x.beginPath(); x.arc(cx, cy, r, 0, TAU); x.stroke();
  x.strokeStyle = css.gust; x.lineWidth = 2.5; x.beginPath(); x.moveTo(cx - r * 0.5, cy); x.lineTo(cx - r * 0.15, cy); x.moveTo(cx + r * 0.15, cy); x.lineTo(cx + r * 0.5, cy); x.stroke();
  x.fillStyle = css.ink; x.font = `500 10.5px ${css.mono}`;
  const hdg = ((90 - yaw) % 360 + 360) % 360;
  x.fillText(`φ ${roll >= 0 ? '+' : '−'}${Math.abs(roll).toFixed(1)}°`, cx * 2 + 4, 14); x.fillText(`θ ${pitch >= 0 ? '+' : '−'}${Math.abs(pitch).toFixed(1)}°`, cx * 2 + 4, 28);
  x.fillText(`hdg ${hdg.toFixed(0).padStart(3, '0')}°`, cx * 2 + 4, 42); x.fillText(`alt ${q.pos[2].toFixed(1)} m`, cx * 2 + 4, 56);
  x.fillText(`v ${Math.hypot(q.vel[0], q.vel[1]).toFixed(1)} m/s`, cx * 2 + 4, 70);
  // motors
  const mx = cx * 2 + 92, bw = 11, bh = H - 34;
  if (mx + 4 * (bw + 5) < W - 70) {
    (q.motors || []).forEach((m, i) => {
      const bx = mx + i * (bw + 5); x.strokeStyle = css.line2; x.strokeRect(bx + 0.5, 8.5, bw, bh);
      x.fillStyle = m > 0.98 ? '#ff7b6e' : css.ink; x.fillRect(bx + 1, 9 + bh * (1 - clamp(m, 0, 1)), bw - 1, bh * clamp(m, 0, 1));
      x.fillStyle = css.muted; x.fillText(String(i + 1), bx + 2, H - 12);
    });
    x.fillStyle = css.muted; x.fillText('motors', mx, H - 1);
  }
  // wind compass
  const wx = W - 34, wy = cy - 4, wr = 26, w = f.wind || [0, 0, 0], sp = Math.hypot(w[0], w[1]);
  x.strokeStyle = css.line2; x.beginPath(); x.arc(wx, wy, wr, 0, TAU); x.stroke();
  x.fillStyle = css.muted; x.fillText('N', wx - 3, wy - wr - 2);
  if (sp > 0.02) { const k = Math.min(1, sp / 5) * wr * 0.95; arrow(x, wx - w[0] / sp * k, wy + w[1] / sp * k, wx + w[0] / sp * k, wy - w[1] / sp * k, css.gust, 2.2, 7); }
  x.fillStyle = css.ink; x.textAlign = 'center'; x.fillText(`${sp.toFixed(1)} m/s`, wx, H - 12); x.fillStyle = css.muted; x.fillText(`w ${w[2] >= 0 ? '+' : '−'}${Math.abs(w[2]).toFixed(1)}`, wx, H - 1); x.textAlign = 'left';
}

// ---------- top-down map (truth): obstacles, objects, home, path, drone, home vector estimate and truth
function map(c, world, trail, f, css, opt) {
  const { x, W, H } = ctx2d(c); if (!world) return;
  const pts = [[world.home[0], world.home[1]], ...world.objects.map((o) => [o.x, o.y])];
  if (f) pts.push([f.quad.pos[0], f.quad.pos[1]]);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [a, b] of pts) { x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, b); y1 = Math.max(y1, b); }
  for (let i = 0; i < trail.n; i++) { const a = trail.xy[2 * i], b = trail.xy[2 * i + 1]; if (a < x0) x0 = a; if (a > x1) x1 = a; if (b < y0) y0 = b; if (b > y1) y1 = b; }
  const span = Math.max(x1 - x0, y1 - y0, 20) * 1.15 + 8, mx = (x0 + x1) / 2, my = (y0 + y1) / 2, s = Math.min(W, H) / span;
  const P = (a, b) => [W / 2 + (a - mx) * s, H / 2 - (b - my) * s];
  x.fillStyle = rgbCss(world.ground ? world.ground.rgb : [0.4, 0.5, 0.3], 0.55); x.fillRect(0, 0, W, H);
  for (const o of world.obstacles) {
    const [px, py] = P(o.x, o.y);
    if (px < -20 || py < -20 || px > W + 20 || py > H + 20) continue;
    x.fillStyle = rgbCss(o.rgb || [0.3, 0.4, 0.3], 0.85);
    if (o.type === 'building') { x.save(); x.translate(px, py); x.rotate(-(o.yaw_deg || 0) * D2R); x.fillRect(-o.w * s / 2, -o.d * s / 2, o.w * s, o.d * s); x.restore(); }
    else { x.beginPath(); x.arc(px, py, Math.max(1.2, o.r * s), 0, TAU); x.fill(); }
  }
  const [hx, hy] = P(world.home[0], world.home[1]);
  x.strokeStyle = '#fff'; x.lineWidth = 1.5; x.beginPath(); x.arc(hx, hy, 6, 0, TAU); x.stroke(); x.fillStyle = '#fff'; x.font = `600 9px ${css.mono}`; x.textAlign = 'center'; x.fillText('H', hx, hy + 3);
  for (const o of world.objects) {
    const [px, py] = P(o.x, o.y), here = !opt.present || opt.present(o);
    x.globalAlpha = here ? 1 : 0.3; x.fillStyle = rgbCss(o.appearance.base); x.strokeStyle = o.role === 'target' ? '#fff' : '#111'; x.lineWidth = 1.5;
    x.beginPath(); x.rect(px - 4, py - 4, 8, 8); x.fill(); x.stroke();
    x.fillStyle = '#fff'; x.fillText(o.role === 'target' ? 'T' : 'D', px, py - 7); x.globalAlpha = 1;
  }
  x.textAlign = 'left';
  if (trail.n > 1) {
    x.strokeStyle = opt.col; x.lineWidth = 1.6; x.beginPath();
    for (let i = 0; i < trail.n; i++) { const [px, py] = P(trail.xy[2 * i], trail.xy[2 * i + 1]); i ? x.lineTo(px, py) : x.moveTo(px, py); }
    x.stroke();
  }
  if (f) {
    const [dx, dy] = P(f.quad.pos[0], f.quad.pos[1]), yaw = f.quad.euler_deg[2] * D2R;
    if (f.pi) {
      arrow(x, dx, dy, dx + f.pi.home_true[0] * s, dy - f.pi.home_true[1] * s, 'rgba(255,255,255,.55)', 1.2, 6, [4, 3]);
      arrow(x, dx, dy, dx + f.pi.home_est[0] * s, dy - f.pi.home_est[1] * s, '#fff', 1.6, 6);
    }
    x.save(); x.translate(dx, dy); x.rotate(-yaw); x.fillStyle = opt.col; x.strokeStyle = '#111'; x.lineWidth = 1;
    x.beginPath(); x.moveTo(8, 0); x.lineTo(-5, 5); x.lineTo(-2, 0); x.lineTo(-5, -5); x.closePath(); x.fill(); x.stroke(); x.restore();
  }
  // scale bar
  const nice = [1, 2, 5, 10, 20, 50, 100].find((v) => v * s > 40) || 100;
  x.strokeStyle = x.fillStyle = '#fff'; x.lineWidth = 1.5; x.beginPath(); x.moveTo(W - 10 - nice * s, H - 10); x.lineTo(W - 10, H - 10); x.stroke();
  x.font = `10px ${css.mono}`; x.textAlign = 'right'; x.fillText(`${nice} m`, W - 10, H - 14); x.textAlign = 'left';
}

// ---------- Results charts (inline SVG strings; theme via CSS classes)
const ARMS = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine'];
const ARMNAME = { ring: 'Ring hub', ring_static: 'Ring, no recurrence', vote: 'Static vote', sum: 'Vector sum', accumulator: 'Leaky accumulator', state_machine: 'State machine' };
const SHAPE = { ring: 'c', ring_static: 'c', vote: 's', sum: 'd', accumulator: 't', state_machine: 'x' };
const cls = (arm) => 'c' + (ROLE[arm] || 'n');
const f1 = (v) => Math.round(v * 10) / 10;
const mk = (arm, X, Y, tip) => {
  const s = SHAPE[arm] || 'c', c = cls(arm), r = 3.6;
  const body = s === 's' ? `<rect class="${c}" x="${f1(X - r)}" y="${f1(Y - r)}" width="${2 * r}" height="${2 * r}"/>`
    : s === 'd' ? `<path class="${c}" d="M${f1(X)},${f1(Y - r * 1.3)}L${f1(X + r * 1.3)},${f1(Y)}L${f1(X)},${f1(Y + r * 1.3)}L${f1(X - r * 1.3)},${f1(Y)}Z"/>`
      : s === 't' ? `<path class="${c}" d="M${f1(X)},${f1(Y - r * 1.3)}L${f1(X + r * 1.2)},${f1(Y + r)}L${f1(X - r * 1.2)},${f1(Y + r)}Z"/>`
        : s === 'x' ? `<path class="${c}" style="fill:none;stroke-width:2" d="M${f1(X - r)},${f1(Y - r)}L${f1(X + r)},${f1(Y + r)}M${f1(X + r)},${f1(Y - r)}L${f1(X - r)},${f1(Y + r)}"/>`
          : `<circle class="${c}" cx="${f1(X)}" cy="${f1(Y)}" r="${r}"/>`;
  return `<g><title>${tip}</title>${body}</g>`;
};
function axes(W, H, xr, yr, xl, yl, m = { l: 40, r: 10, t: 22, b: 30 }) {
  const X = (v) => m.l + ((v - xr[0]) / (xr[1] - xr[0])) * (W - m.l - m.r), Y = (v) => H - m.b - ((v - yr[0]) / (yr[1] - yr[0])) * (H - m.t - m.b);
  const ticks = (r) => { const st = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20].find((s) => (r[1] - r[0]) / s <= 5); const o = []; for (let v = Math.ceil(r[0] / st) * st; v <= r[1] + 1e-9; v += st) o.push(+v.toFixed(3)); return o; };
  let s = '';
  for (const t of ticks(xr)) s += `<line class="grid" x1="${f1(X(t))}" x2="${f1(X(t))}" y1="${m.t}" y2="${H - m.b}"/><text x="${f1(X(t))}" y="${H - m.b + 13}" text-anchor="middle">${t}</text>`;
  for (const t of ticks(yr)) s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${f1(Y(t))}" y2="${f1(Y(t))}"/><text x="${m.l - 5}" y="${f1(Y(t)) + 3.5}" text-anchor="end">${t}</text>`;
  s += `<rect class="frame" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">${xl}</text><text transform="translate(10,${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
  return { X, Y, s };
}
const svg = (W, H, title, body) => `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${title}"><text x="4" y="13" style="fill:var(--ink);font-weight:600">${title}</text>${body}</svg>`;
const legCols = (W) => (W >= 520 ? 3 : 2);
const legTop = (n, W) => 30 + Math.ceil(n / legCols(W)) * 13 + 6;
const legend = (arms, W) => arms.map((a, i) => { const nc = legCols(W), x = 44 + (i % nc) * ((W - 50) / nc), y = 30 + Math.floor(i / nc) * 13; return mk(a, x, y - 3.5, ARMNAME[a]) + `<text x="${x + 7}" y="${y}">${ARMNAME[a]}</text>`; }).join('');

function chartFrontier(S, W = 420, H = 260) {
  const F = S && S.stubbornness_frontier; if (!F) return '';
  const arms = ARMS.filter((a) => F[a]); let xm = 1, ym = 1;
  for (const a of arms) for (const p of F[a]) { xm = Math.max(xm, (p.hold_ci || [0, p.hold_s])[1]); ym = Math.max(ym, (p.release_ci || [0, p.release_s])[1]); }
  const A = axes(W, H, [0, Math.ceil(xm)], [0, Math.ceil(ym)], 'hidden target held (s), higher = better memory', 'time to release decoy (s)', { l: 40, r: 10, t: legTop(arms.length, W), b: 30 });
  let s = A.s;
  for (const a of arms) {
    const pts = F[a];
    for (const p of pts) {
      if (p.hold_ci) s += `<line class="${cls(a)} wh" x1="${f1(A.X(p.hold_ci[0]))}" x2="${f1(A.X(p.hold_ci[1]))}" y1="${f1(A.Y(p.release_s))}" y2="${f1(A.Y(p.release_s))}"/>`;
      if (p.release_ci) s += `<line class="${cls(a)} wh" x1="${f1(A.X(p.hold_s))}" x2="${f1(A.X(p.hold_s))}" y1="${f1(A.Y(p.release_ci[0]))}" y2="${f1(A.Y(p.release_ci[1]))}"/>`;
    }
    if (pts.length > 1) s += `<path class="${cls(a)} ln" d="${pts.map((p, i) => `${i ? 'L' : 'M'}${f1(A.X(p.hold_s))},${f1(A.Y(p.release_s))}`).join('')}"/>`;
    for (const p of pts) s += mk(a, A.X(p.hold_s), A.Y(p.release_s), `${ARMNAME[a]}, knob ${p.gain_or_knob}: hold ${p.hold_s.toFixed(2)} s, release ${p.release_s.toFixed(2)} s`);
    if (a === 'ring' && pts.length > 1) for (const p of pts) s += `<text class="lbl" x="${f1(A.X(p.hold_s) + 5)}" y="${f1(A.Y(p.release_s) - 5)}">g=${p.gain_or_knob}</text>`;
  }
  return svg(W, H, 'Stubbornness trade-off (H3)', s + legend(arms, W));
}
function chartSeverity(S, W = 420, H = 260) {
  const D = S && S.h1_success_vs_severity; if (!D) return '';
  const arms = ARMS.filter((a) => D[a]);
  const A = axes(W, H, [0, 1], [0, 1], 'fault severity (0 = none, 1 = worst)', 'mission success rate', { l: 40, r: 10, t: legTop(arms.length, W), b: 30 });
  let s = A.s;
  for (const a of arms) {
    const p = D[a];
    if (p[0] && p[0].ci) s += `<path class="${cls(a)} bd" d="${p.map((q, i) => `${i ? 'L' : 'M'}${f1(A.X(q.severity))},${f1(A.Y(q.ci[1]))}`).join('')}${p.slice().reverse().map((q) => `L${f1(A.X(q.severity))},${f1(A.Y(q.ci[0]))}`).join('')}Z"/>`;
    s += `<path class="${cls(a)} ln" d="${p.map((q, i) => `${i ? 'L' : 'M'}${f1(A.X(q.severity))},${f1(A.Y(q.success))}`).join('')}"/>`;
    for (const q of p) s += mk(a, A.X(q.severity), A.Y(q.success), `${ARMNAME[a]}, severity ${q.severity}: ${(100 * q.success).toFixed(0)}%`);
  }
  return svg(W, H, 'Success vs fault severity (H1)', s + legend(arms, W));
}
function chartAuroc(S, W = 420, H = 260) {
  const D = S && S.recog_auroc; if (!D) return '';
  const meths = Object.keys(D), NAME = { mb: 'Mushroom body', pixels: 'Pixels', colour_hist: 'Colour hist.', dinov2: 'DINOv2' };
  const C = { mb: 'ca', dinov2: 'cb' };
  const A = axes(W, H, [0, meths.length], [0.4, 1], '', 'swap AUROC (0.5 = chance)', { l: 40, r: 10, t: 36, b: 30 });
  let s = A.s.replace(/<text x="[\d.]+" y="[\d.]+" text-anchor="middle">\d+(\.\d+)?<\/text>/g, '');
  s += `<line class="cn" style="stroke-dasharray:3 3" x1="${A.X(0)}" x2="${A.X(meths.length)}" y1="${A.Y(0.5)}" y2="${A.Y(0.5)}"/>`;
  meths.forEach((m, i) => {
    const c = C[m] || 'cn';
    [['no_flight', -0.15, true], ['flight', 0.15, false]].forEach(([k, dx, open]) => {
      const v = D[m][k]; if (!v) return; const X = A.X(i + 0.5 + dx);
      s += `<line class="${c} wh" x1="${f1(X)}" x2="${f1(X)}" y1="${f1(A.Y(v.lo))}" y2="${f1(A.Y(v.hi))}"/>`;
      s += `<g><title>${NAME[m] || m}, ${k.replace('_', ' ')}: ${v.mean.toFixed(2)} [${v.lo.toFixed(2)}, ${v.hi.toFixed(2)}]</title><circle class="${c}" cx="${f1(X)}" cy="${f1(A.Y(v.mean))}" r="3.8" ${open ? 'style="fill:var(--panel)"' : ''}/></g>`;
    });
    s += `<text x="${f1(A.X(i + 0.5))}" y="${H - 17}" text-anchor="middle">${NAME[m] || m}</text>`;
  });
  s += `<circle class="cn" cx="48" cy="26" r="3.6" style="fill:var(--panel)"/><text x="56" y="29.5">no learning flight</text><circle class="cn" cx="170" cy="26" r="3.6"/><text x="178" y="29.5">10 s learning flight</text>`;
  return svg(W, H, 'Re-finding the target (H4)', s);
}

// ---------- Study 2 (v2) charts: summary.v2 (scripts/export_app_summary.py, from results/mission_v2.json)
const V2NAME = { ring: 'Study 2 brain (ring)', ring_static: 'Ring, no recurrence', vote: 'Static vote', sum: 'Vector sum', accumulator: 'Leaky accumulator', state_machine: 'State machine', v1: 'Study 1 brain' };
const FNAME = { none: 'none', occlusion: 'occlusion', decoy: 'decoy', compass_half: 'compass ½', eye_columns: 'eye cols', false_sun: 'false sun', speed_scale: 'speed', gust: 'gusts' };
// Study 1 brain: hollow circle with a diagonal hatch line (a different success definition, see chartV2Success)
const v2mk = (a, X, Y, tip) => (a === 'v1' ? `<g><title>${tip}</title><circle class="cn" cx="${f1(X)}" cy="${f1(Y)}" r="3.6" style="fill:var(--panel);stroke-width:1.4"/><line class="cn" x1="${f1(X - 2.5)}" y1="${f1(Y + 2.5)}" x2="${f1(X + 2.5)}" y2="${f1(Y - 2.5)}" style="stroke-width:1.1"/></g>` : mk(a, X, Y, tip));
const v2legend = (arms, W, name = (a) => V2NAME[a]) => arms.map((a, i) => { const nc = legCols(W), x = 44 + (i % nc) * ((W - 50) / nc), y = 30 + Math.floor(i / nc) * 13; return v2mk(a, x, y - 3.5, name(a)) + `<text x="${x + 7}" y="${y}">${name(a)}</text>`; }).join('');
const xcats = (A, cats, H, m) => cats.map((c, i) => `<text transform="translate(${f1(A.X(i + 0.5) + 3)},${H - m.b + 9}) rotate(-35)" text-anchor="end">${FNAME[c] || c}</text>`).join('');
const noXTicks = (s) => s.replace(/<text x="[\d.]+" y="[\d.]+" text-anchor="middle">-?\d+(\.\d+)?<\/text>/g, '');
/** success per condition at the pre-registered middle severity (H9 iii levels), all six v2 arbiters and the Study 1 brain */
function chartV2Success(S, W = 420, H = 260) {
  const M = S && S.v2 && S.v2.m, H9 = S && S.v2 && S.v2.h9; if (!M || !M.curves || !H9) return '';
  const arms = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine', 'v1'].filter((a) => (a === 'v1' ? M.v1own || M.curves.v1 : M.curves[a]));
  const cats = ['none', ...M.faults];
  const name = (a) => (a === 'v1' ? (M.v1own ? 'Study 1 brain (own 2 m success, no landing)' : V2NAME.v1) : V2NAME[a]);
  // the Study 1 brain has no landing module: it is plotted on its own success definition (target found, home within
  // 2 m, no collision; summary v2.m.v1own), as in both papers, not on the 0.5 m pad-landing definition (about 0 everywhere)
  const own = M.v1own, src = (a) => (a === 'v1' && own ? { none: own.success_none, nci: own.none_ci, curves: own.curves } : { none: M.success_none[a], nci: M.none_ci[a], curves: M.curves[a] });
  const pt = (a, c) => {
    const d = src(a);
    if (c === 'none') return { v: d.none, ci: d.nci, lvl: '' };
    const lv = H9.iii[c].level, i = M.levels[c].findIndex((x) => (isNaN(+x) ? x === String(lv) : +x === +lv));
    const q = d.curves && d.curves[c] && d.curves[c][i]; return q ? { v: q.success, ci: q.ci, lvl: M.levels[c][i] } : null;
  };
  const m = { l: 40, r: 10, t: legTop(arms.length, W), b: 46 };
  const A = axes(W, H, [0, cats.length], [0, 1], '', 'full success', m);
  let s = noXTicks(A.s) + xcats(A, cats, H, m);
  cats.forEach((c, i) => arms.forEach((a, k) => {
    const p = pt(a, c); if (!p || !Number.isFinite(p.v)) return;
    const X = A.X(i + 0.5 + (k - (arms.length - 1) / 2) * 0.11);
    if (p.ci) s += `<line class="${a === 'v1' ? 'cn' : cls(a)} wh" x1="${f1(X)}" x2="${f1(X)}" y1="${f1(A.Y(p.ci[0]))}" y2="${f1(A.Y(p.ci[1]))}"/>`;
    s += v2mk(a, X, A.Y(p.v), `${name(a)}, ${FNAME[c] || c}${p.lvl ? ' ' + p.lvl : ''}: ${(100 * p.v).toFixed(0)}% (95% CI ${(100 * p.ci[0]).toFixed(0)}–${(100 * p.ci[1]).toFixed(0)}%, n = 300)`);
  }));
  return svg(W, H, 'Success by condition, middle severity (Study 2)', s + v2legend(arms, W, name));
}
/** paired drop in full-success AUC when one part of the Study 2 brain is removed or changed (v2 ring minus arm) */
function chartV2Ablation(S, W = 420, H = 260) {
  const M = S && S.v2 && S.v2.m; if (!M || !M.abl) return '';
  const rows = [...Object.entries(M.abl).map(([k, v]) => [{ heading: 'no heading ring', home: 'no home snapshot', landing: 'no landing', front: 'no insect front end', inspect: 'no inspection rule', gated_arbiter: 'no evidence gate' }[k.replace('v2_ring_minus_', '')] || k, v]),
    ...(M.extra ? [['Kalman heading', M.extra.kf], ['poor tip', M.extra.poor_tip], ['no tip', M.extra.no_tip]] : []),
    ...(M.lf ? [['no learning flight', M.lf]] : [])].filter(([, v]) => v && Number.isFinite(v.est));
  const m = { l: 118, r: 14, t: 24, b: 30 }, n = rows.length, hi = Math.max(0.1, ...rows.map(([, v]) => v.hi));
  const A = axes(W, H, [Math.min(0, ...rows.map(([, v]) => v.lo)), Math.ceil(hi * 10) / 10], [0, n], 'drop in full-success AUC (paired, 95% CI)', '', m);
  let s = A.s.replace(/<text x="[\d.]+" y="[\d.]+" text-anchor="end">[\d.]+<\/text>/g, '').replace(/<line class="grid" x1="[\d.]+" x2="[\d.]+" y1="[\d.]+" y2="[\d.]+"\/>/g, (g) => (/x1="(\d+(\.\d+)?)" x2="\1"/.test(g) ? g : ''));
  s += `<line class="cn" x1="${f1(A.X(0))}" x2="${f1(A.X(0))}" y1="${m.t}" y2="${H - m.b}"/>`;
  rows.forEach(([lab, v], i) => {
    const Y = A.Y(n - i - 0.5), c = /no tip|poor tip|Kalman/.test(lab) ? 'cn' : 'ca';
    s += `<g><title>${lab}: ${v.est.toFixed(3)} [${v.lo.toFixed(3)}, ${v.hi.toFixed(3)}]</title><rect class="${c}" style="opacity:.75" x="${f1(Math.min(A.X(0), A.X(v.est)))}" y="${f1(Y - 5)}" width="${f1(Math.abs(A.X(v.est) - A.X(0)))}" height="10"/>`
      + `<line class="${c} wh" x1="${f1(A.X(v.lo))}" x2="${f1(A.X(v.hi))}" y1="${f1(Y)}" y2="${f1(Y)}" style="stroke:var(--ink)"/></g><text x="${m.l - 6}" y="${f1(Y + 3.5)}" text-anchor="end">${lab}</text>`;
  });
  return svg(W, H, 'What each part is worth (Study 2)', s);
}
/** landing on the home pad: share of runs landed on the pad (error <= 0.5 m) per condition, with the median error */
function chartV2Landing(S, W = 420, H = 260) {
  const M = S && S.v2 && S.v2.m; if (!M || !M.landing || !M.landing.ring) return '';
  const L = M.landing, cats = ['none', ...(M.faults || Object.keys(L.ring).filter((k) => k !== 'none'))];
  const arms = [['ring', 'ca', 'Study 2 brain'], ['v2_ring_minus_home', 'cn', 'no home snapshot'], ['v1', 'cn', 'Study 1 brain: cannot land (no landing module)', 'Study 1: cannot land (no landing module)']].filter(([a]) => L[a]);
  const m = { l: 40, r: 10, t: 48, b: 46 };
  const A = axes(W, H, [0, cats.length], [0, 1], '', 'landed on the pad (≤ 0.5 m)', m);
  let s = noXTicks(A.s) + xcats(A, cats, H, m);
  const bw = Math.min(10, (A.X(1) - A.X(0)) / (arms.length + 1.5));
  cats.forEach((c, i) => arms.forEach(([a, cl, nm], k) => {
    const v = L[a][c]; if (!v || !Number.isFinite(v.landed_on_pad)) return;
    const X = A.X(i + 0.5) + (k - (arms.length - 1) / 2) * (bw + 2), y = A.Y(v.landed_on_pad);
    const st = a === 'v1' ? 'fill:var(--panel);stroke-width:1.2' : a === 'ring' ? '' : 'opacity:.55';
    s += `<g><title>${nm}, ${FNAME[c] || c}: ${(100 * v.landed_on_pad).toFixed(0)}% on the pad; median landing error ${Number.isFinite(v.landing_error_median_m) ? (100 * v.landing_error_median_m).toFixed(1) + ' cm' : '–'}; median touchdown ${Number.isFinite(v.touchdown_speed_median_mps) ? v.touchdown_speed_median_mps.toFixed(2) + ' m/s' : '– (no touchdown)'}</title><rect class="${cl}" style="${st}" x="${f1(X - bw / 2)}" y="${f1(y)}" width="${f1(bw)}" height="${f1(Math.max(0.5, A.Y(0) - y))}"/></g>`;
  }));
  s += arms.map(([a, cl, nm, short], k) => `<rect class="${cl}" style="${a === 'v1' ? 'fill:var(--panel)' : a === 'ring' ? '' : 'opacity:.55'}" x="${f1(44 + (k % 2) * (W - 50) / 2)}" y="${22 + 12 * Math.floor(k / 2)}" width="8" height="8"/><text x="${f1(56 + (k % 2) * (W - 50) / 2)}" y="${29.5 + 12 * Math.floor(k / 2)}"${short ? ' style="font-size:9.5px"' : ''}>${short || nm}</text>`).join('');
  return svg(W, H, 'Landing on the home pad (Study 2)', s);
}

window.Panels = { SM_STATES, eye, pic, fam, hub, homev, att, map, chartFrontier, chartSeverity, chartAuroc, chartV2Success, chartV2Ablation, chartV2Landing, ROLE, roleCol, ARMNAME, eyeBytes };
})();
