// Mutant brain drone: on-screen display (OSD) drawn on a 2D canvas over the 3D view. Classic script; defines
// window.makeHUD(canvas, host). draw(frame, opt) once per animation frame; fields a frame lacks are simply not drawn.
// Colours: neutral = simulator truth; opt.brain (--a) = the brain's own estimates and commands.
// Frame fields used (C6): quad {pos, vel, euler_deg, motors}, wind, pi {home_est, home_true, heading_est_deg},
// ring.heading_cmd_deg, arbiter.commit, familiarity, phase, faults_active, outcome, home (v2 landing state). Newer optional
// frame.hud (alt, set-point, speeds, headings, home distances, energy, heading confidence, home familiarity, landing
// stage) is used when present and computed from the fields above otherwise (recorded replays lack it).
'use strict';
window.makeHUD = function (cv, host) {
  const ctx = cv.getContext('2d');
  const D2R = Math.PI / 180, clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const wrap360 = (a) => ((a % 360) + 360) % 360, wrap180 = (a) => ((a + 540) % 360) - 180;
  const fin = (v) => typeof v === 'number' && Number.isFinite(v);
  const pick = (f, paths) => { for (const p of paths) { let v = f; for (const k of p.split('.')) v = v == null ? undefined : v[k]; if (v != null) return v; } return undefined; };
  let W = 0, H = 0, on = true, eAcc = 0, eT = null;
  const TRUTH = 'rgba(244,246,248,0.92)', DIM = 'rgba(244,246,248,0.55)', HALO = 'rgba(8,10,12,0.55)';

  function size() {
    const w = host.clientWidth, h = host.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
    if (w !== W || h !== H || cv.width !== Math.round(w * dpr)) { W = w; H = h; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(cv.width / Math.max(W, 1), 0, 0, cv.height / Math.max(H, 1), 0, 0);
  }
  // ---- primitives (every stroke and glyph gets a dark halo so it reads over sky, grass and fog)
  let fs = 11;
  const font = (k = 1, w = 500) => { ctx.font = `${w} ${Math.round(fs * k)}px "IBM Plex Mono", ui-monospace, monospace`; };
  function text(s, x, y, col = TRUTH, align = 'left', base = 'middle') {
    ctx.textAlign = align; ctx.textBaseline = base; ctx.lineJoin = 'round';
    ctx.strokeStyle = HALO; ctx.lineWidth = 3; ctx.strokeText(s, x, y); ctx.fillStyle = col; ctx.fillText(s, x, y);
    return ctx.measureText(s).width;
  }
  function stroke(path, col = TRUTH, w = 1.25) {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = HALO; ctx.lineWidth = w + 2; ctx.beginPath(); path(); ctx.stroke();
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); path(); ctx.stroke();
  }
  function fillP(path, col) { ctx.fillStyle = HALO; ctx.beginPath(); path(); ctx.lineWidth = 2; ctx.strokeStyle = HALO; ctx.stroke(); ctx.fillStyle = col; ctx.beginPath(); path(); ctx.fill(); }
  // ---- layout: every block claims a rectangle; a block that would overlap a claimed one (or an overlay) is skipped
  let taken = [];
  const hit = (a, b) => a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
  const claim = (r) => { if (r[0] < 0 || r[1] < 0 || r[0] + r[2] > W || r[1] + r[3] > H || taken.some((t) => hit(r, t))) return false; taken.push(r); return true; };
  // text block: lines = [[[str, col], ...], ...]; anchor 'tl' | 'tr' | 'br' | 'bl'
  function block(lines, x, y, anchor) {
    font(0.95); const lh = Math.round(fs * 1.38);
    const widths = lines.map((ln) => ln.reduce((a, [s]) => a + ctx.measureText(s).width, 0));
    const bw = Math.max(0, ...widths) + 4, bh = lines.length * lh;
    const bx = anchor[1] === 'r' ? x - bw : x, by = anchor[0] === 'b' ? y - bh : y;
    if (!lines.length || !claim([bx - 2, by - 2, bw + 4, bh + 4])) return false;
    lines.forEach((ln, i) => { let cx = anchor[1] === 'r' ? x - widths[i] : x; for (const [s, c] of ln) cx += text(s, cx, by + lh * (i + 0.5), c || TRUTH); });
    return true;
  }

  const cy0 = (h) => h * 0.46;
  function draw(f, opt) {
    size(); ctx.clearRect(0, 0, W, H);
    if (!on || !f || !f.quad) return;
    const brain = opt.brain || '#f08a3c', flag = opt.flag || '#d9b45a', gust = opt.gust || '#ffd23f';
    const s = clamp(Math.min(W, H) / 560, 0.85, 1.25), narrow = W < 560; fs = 11 * s;
    taken = (opt.avoid || []).map((r) => [r[0] - 4, r[1] - 4, r[2] + 8, r[3] + 8]);
    const q = f.quad, eu = q.euler_deg || [0, 0, 0], roll = eu[0], up = -eu[1];       // FLU body: +pitch = nose down
    const hu = f.hud || {};   // frame.hud (app/sim/mission.js): angles there are ENU (CCW from east), as everywhere in the sim
    const hdgT = wrap360(90 - (pick(f, ['hud.heading_true_deg']) ?? eu[2]));            // compass, 0 = north, clockwise
    const estRaw = pick(f, ['hud.heading_est_deg', 'pi.heading_est_deg']), hdgE = fin(estRaw) ? wrap360(90 - estRaw) : null;
    const cmdRaw = pick(f, ['hud.heading_cmd_deg', 'ring.heading_cmd_deg']);
    // the command lives in the brain's compass frame: drawn where the drone will actually point once its estimate reaches it
    const hdgC = fin(cmdRaw) && fin(estRaw) ? wrap360(hdgT - wrap180(cmdRaw - estRaw)) : null;
    const v = q.vel || null, w = f.wind || [0, 0, 0];
    const gs = pick(f, ['hud.ground_speed_mps']) ?? (v ? Math.hypot(v[0], v[1]) : null);
    const as = pick(f, ['hud.air_speed_mps']) ?? (v ? Math.hypot(v[0] - w[0], v[1] - w[1], v[2] - w[2]) : null);
    const alt = pick(f, ['hud.alt_m']) ?? q.pos[2], vz = pick(f, ['hud.vz_mps']) ?? (v ? v[2] : null);
    const altSp = f.hud ? (fin(hu.alt_sp_m) ? hu.alt_sp_m : null) : f.phase === 'done' || f.phase === 'learning' ? null : 2.5;   // replays: the v1 cruise altitude

    // ---------- attitude: horizon, pitch ladder, roll scale (centre)
    const cx = W / 2, cy = H * 0.46, R = clamp(Math.min(Math.min(W, H) * 0.19, (cy0(H) - ((opt.topY || 0) + 14 + 46 * s)) / 1.32), 30, 400), k = R / 25;
    taken.push([cx - R * 1.3, cy - R * 1.32, R * 2.6, R * 2.37]);   // the attitude display always draws (overlays sit above it)
    {
      ctx.save(); ctx.translate(cx, cy); ctx.beginPath(); ctx.ellipse(0, 0, R * 1.3, R * 0.9, 0, 0, 7); ctx.clip();
      ctx.rotate(-roll * D2R); ctx.translate(0, up * k);
      stroke(() => { ctx.moveTo(-R * 1.6, 0); ctx.lineTo(-R * 0.32, 0); ctx.moveTo(R * 0.32, 0); ctx.lineTo(R * 1.6, 0); }, TRUTH, 1.4);
      font(0.82);
      for (let p = -40; p <= 40; p += 5) {
        if (!p) continue;
        const y = -p * k, hw = p % 10 ? R * 0.14 : R * 0.3, neg = p < 0;
        if (Math.abs(y + up * k) > R * 0.95) continue;
        ctx.setLineDash(neg ? [4, 3] : []);
        stroke(() => { ctx.moveTo(-hw, y); ctx.lineTo(-R * 0.12, y); ctx.moveTo(R * 0.12, y); ctx.lineTo(hw, y); }, DIM, 1);
        ctx.setLineDash([]);
        if (!(p % 10) && !narrow) { text(String(Math.abs(p)), -hw - 4, y, DIM, 'right'); text(String(Math.abs(p)), hw + 4, y, DIM, 'left'); }
      }
      ctx.restore();
      // fixed aircraft symbol
      stroke(() => { ctx.moveTo(cx - R * 0.26, cy); ctx.lineTo(cx - R * 0.1, cy); ctx.lineTo(cx - R * 0.05, cy + R * 0.06); ctx.moveTo(cx + R * 0.26, cy); ctx.lineTo(cx + R * 0.1, cy); ctx.lineTo(cx + R * 0.05, cy + R * 0.06); ctx.moveTo(cx + 1.5, cy); ctx.arc(cx, cy, 1.5, 0, 7); }, TRUTH, 1.6);
      // roll scale (fixed arc, pointer turns with the bank)
      const rr = R * 1.22;
      stroke(() => { ctx.arc(cx, cy, rr, (-90 - 60) * D2R, (-90 + 60) * D2R); for (const a of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) { const L = a % 30 ? 5 : 9, t = (a - 90) * D2R; ctx.moveTo(cx + rr * Math.cos(t), cy + rr * Math.sin(t)); ctx.lineTo(cx + (rr + L) * Math.cos(t), cy + (rr + L) * Math.sin(t)); } }, DIM, 1);
      const rp = (clamp(roll, -60, 60) - 90) * D2R, px = cx + (rr - 2) * Math.cos(rp), py = cy + (rr - 2) * Math.sin(rp);
      fillP(() => { ctx.moveTo(px, py); ctx.lineTo(px - 5 * Math.cos(rp) + 4 * Math.sin(rp), py - 5 * Math.sin(rp) - 4 * Math.cos(rp)); ctx.lineTo(px - 5 * Math.cos(rp) - 4 * Math.sin(rp), py - 5 * Math.sin(rp) + 4 * Math.cos(rp)); ctx.closePath(); }, TRUTH);
      if (!narrow) { font(0.82); text(`${roll >= 0 ? 'R' : 'L'}${Math.abs(roll).toFixed(0)}° P${up >= 0 ? '+' : '−'}${Math.abs(up).toFixed(0)}°`, cx, cy + R * 0.98, DIM, 'center'); }
    }

    // ---------- heading tape (top centre): truth under the fixed index; brain estimate caret and command bug in orange
    const tw = Math.min(W * (narrow ? 0.62 : 0.44), 440), span = narrow ? 60 : 80, ppd = tw / span, ty = Math.round((opt.topY || 0) + 14 + 6 * s), tx = cx - tw / 2;
    if (claim([tx - 6, ty - 8, tw + 12, 40 * s + 10])) {
      font(0.82);
      stroke(() => { for (let d = Math.ceil((hdgT - span / 2) / 5) * 5; d <= hdgT + span / 2; d += 5) { const x = cx + (d - hdgT) * ppd, L = d % 10 ? 4 : 7; ctx.moveTo(x, ty); ctx.lineTo(x, ty + L); } }, DIM, 1);
      for (let d = Math.ceil((hdgT - span / 2) / 30) * 30; d <= hdgT + span / 2; d += 30) {
        const x = cx + (d - hdgT) * ppd, a = wrap360(d), lbl = { 0: 'N', 90: 'E', 180: 'S', 270: 'W' }[a] ?? String(a / 10).padStart(2, '0');
        if (Math.abs(x - cx) > 22 * s) text(lbl, x, ty + 15 * s, a % 90 ? DIM : TRUTH, 'center');
      }
      const caret = (h, col, up_) => { const dx = wrap180(h - hdgT) * ppd, x = cx + clamp(dx, -tw / 2, tw / 2); fillP(() => { const y = up_ ? ty - 1 : ty + 1; ctx.moveTo(x, y); ctx.lineTo(x - 5, up_ ? y - 7 : y + 7); ctx.lineTo(x + 5, up_ ? y - 7 : y + 7); ctx.closePath(); }, col); return x; };
      if (hdgC != null) { const dx = clamp(wrap180(hdgC - hdgT) * ppd, -tw / 2, tw / 2); stroke(() => { ctx.moveTo(cx + dx - 6, ty - 8); ctx.lineTo(cx + dx - 6, ty - 2); ctx.lineTo(cx + dx - 2, ty - 2); ctx.lineTo(cx + dx, ty - 5); ctx.lineTo(cx + dx + 2, ty - 2); ctx.lineTo(cx + dx + 6, ty - 2); ctx.lineTo(cx + dx + 6, ty - 8); }, brain, 1.5); }
      if (hdgE != null) caret(hdgE, brain, false);
      stroke(() => { ctx.moveTo(cx, ty - 9); ctx.lineTo(cx, ty + 8); }, TRUTH, 1.6);
      font(1, 600); const hs = String(Math.round(hdgT) % 360).padStart(3, '0');
      text(hs + '°', cx, ty + 28 * s, TRUTH, 'center');
      if (hdgE != null) { font(0.82); text('est ' + String(Math.round(hdgE) % 360).padStart(3, '0') + '°', cx + 34 * s, ty + 28 * s, brain, 'left'); }
      if (hdgC != null && !narrow) { font(0.82); text('cmd ' + String(Math.round(hdgC) % 360).padStart(3, '0') + '°', cx - 34 * s, ty + 28 * s, brain, 'right'); }
    }

    // ---------- tapes: speed (left), altitude with set-point bug and vertical speed (right)
    const th0 = Math.min(R * 2.3, H * 0.5), gap = Math.min(R * 2.1, W * 0.3), tW = 46 * s;
    let th = th0, tY = cy - th / 2;
    function tape(x, side, val, lo, hi, step, lbl, unit, extra) {
      if (!fin(val)) return false;
      let ok = false;
      for (const k of [1, 0.8, 0.62, 0.5]) {   // shorten the tape until it clears the overlays (e.g. the map)
        th = th0 * k; tY = cy - th / 2;
        const r = side < 0 ? [x - tW, tY - 16 * s, tW + 4, th + 34 * s] : [x - 4, tY - 16 * s, tW + 22 * s, th + 34 * s];
        if (claim(r)) { ok = true; lo *= k; hi *= k; break; }
      }
      if (!ok) return false;
      const ppm = th / (hi - lo), Y = (u) => cy - (u - val) * ppm;
      ctx.save(); ctx.beginPath(); ctx.rect(x - tW - 30, tY, tW + 90, th); ctx.clip();
      stroke(() => { ctx.moveTo(x, tY); ctx.lineTo(x, tY + th); for (let u = Math.ceil((val + lo) / step) * step; u <= val + hi; u += step) { const y = Y(u), L = Math.abs(u / (2 * step) - Math.round(u / (2 * step))) < 1e-6 ? 8 : 4; ctx.moveTo(x, y); ctx.lineTo(x + side * L, y); } }, DIM, 1);
      font(0.82);
      for (let u = Math.ceil((val + lo) / (2 * step)) * 2 * step; u <= val + hi; u += 2 * step) if (Math.abs(Y(u) - cy) > 10 * s) text(u.toFixed(step < 1 ? (step * 2 < 1 ? 1 : 0) : 0), x + side * 11, Y(u), DIM, side < 0 ? 'right' : 'left');
      if (extra) extra(Y);
      ctx.restore();
      font(1, 600); const sv = val.toFixed(unit === 'm' ? 2 : 1), bw = ctx.measureText(sv).width + 10, bx = side < 0 ? x - bw - 6 : x + 6;
      stroke(() => { ctx.rect(bx, cy - 9 * s, bw, 18 * s); }, TRUTH, 1.1);
      text(sv, bx + bw / 2, cy, TRUTH, 'center');
      font(0.78); text(lbl, x + side * 4, tY - 8 * s, DIM, side < 0 ? 'right' : 'left');
      return true;
    }
    const gsTape = tape(cx - gap, -1, gs, -2, 2, 0.5, 'GS m/s', 'm/s');
    if (fin(as) && gsTape && W >= 420) { font(0.82); const r = [cx - gap - tW, tY + th + 20 * s, tW + 4, 14 * s]; if (claim(r)) text('AS ' + as.toFixed(1), cx - gap - 4, tY + th + 26 * s, DIM, 'right'); }
    const altTape = tape(cx + gap, 1, alt, -2, 2, 0.5, 'ALT m', 'm', (Y) => {
      if (fin(altSp)) { const y = clamp(Y(altSp), tY + 4, tY + th - 4); stroke(() => { ctx.moveTo(cx + gap - 1, y - 5); ctx.lineTo(cx + gap - 7, y - 5); ctx.lineTo(cx + gap - 7, y + 5); ctx.lineTo(cx + gap - 1, y + 5); ctx.lineTo(cx + gap - 4, y); ctx.closePath(); }, brain, 1.4); }
    });
    if (fin(vz) && fin(alt) && taken.some((r) => r[0] === cx + gap - 4)) {   // vertical-speed bar beside the altitude tape
      const x = cx + gap + tW + 12 * s, L = clamp(vz / 2, -1, 1) * th / 2;
      stroke(() => { ctx.moveTo(x - 3, cy); ctx.lineTo(x + 3, cy); ctx.moveTo(x, cy - th / 2); ctx.lineTo(x, cy + th / 2); }, DIM, 0.8);
      stroke(() => { ctx.moveTo(x, cy); ctx.lineTo(x, cy - L); }, TRUTH, 3);
      font(0.78); text((vz >= 0 ? '+' : '−') + Math.abs(vz).toFixed(1), x, cy + th / 2 + 9 * s, DIM, 'center');
    }

    // ---------- corner readouts
    const pad = 8;
    const ph = String(f.phase || '–').toUpperCase();
    const tl = [[['T ', DIM], [(f.t ?? 0).toFixed(1).padStart(6) + ' s']], [['PHASE ', DIM], [ph]]];
    if (hu.version) tl.push([['BRAIN ', DIM], ['v' + hu.version]]);
    if (f.outcome) tl.push([[(f.outcome.success_target ? 'FOUND TARGET' : f.outcome.wrong_object_stop ? 'STOPPED AT DECOY' : 'TARGET NOT FOUND') + ' · ' + (f.outcome.success_home ? 'HOME' : 'NOT HOME'), flag]]);
    for (const fa of f.faults_active || []) tl.push([['FAULT ', DIM], [fa.replace(/_/g, ' '), flag]]);
    block(tl, pad, pad + (opt.topLeft || 0), 'tl');

    const tr = [];
    const hdE = pick(f, ['hud.home_dist_est_m']) ?? (f.pi ? Math.hypot(f.pi.home_est[0], f.pi.home_est[1]) : null);
    const hdT = pick(f, ['hud.home_dist_true_m']) ?? (f.pi ? Math.hypot(f.pi.home_true[0], f.pi.home_true[1]) : null);
    if (fin(hdE) || fin(hdT)) tr.push([['HOME ', DIM], [fin(hdE) ? hdE.toFixed(1) : '–', brain], [' | ', DIM], [fin(hdT) ? hdT.toFixed(1) : '–'], [' m', DIM]]);
    const ws = Math.hypot(w[0], w[1]);
    tr.push([['WIND    ', DIM], [ws.toFixed(1)], [' m/s', DIM]]);
    if (!gsTape && fin(gs)) tr.push([['GS ', DIM], [gs.toFixed(1)], [fin(as) ? ' AS ' : '', DIM], [fin(as) ? as.toFixed(1) : ''], [' m/s', DIM]]);   // tapes hidden: numbers instead
    if (!altTape && fin(alt)) tr.push([['ALT ', DIM], [alt.toFixed(2)], [fin(altSp) ? ' → ' : '', DIM], [fin(altSp) ? altSp.toFixed(1) : '', brain], [' m', DIM]]);
    let en = pick(f, ['hud.energy_so_far']), enLbl = 'ENERGY ';
    if (!fin(en) && q.motors) {   // fallback: integrate the normalised motor commands (proxy, resets with the timeline)
      if (eT == null || f.t < eT) { eAcc = 0; eT = f.t; }
      eAcc += q.motors.reduce((a, m) => a + Math.max(m, 0) ** 1.5, 0) * Math.min(f.t - eT, 1); eT = f.t; en = eAcc; enLbl = 'ENERGY* ';
    }
    if (fin(en)) tr.push([[enLbl, DIM], [en.toFixed(1)], [enLbl === 'ENERGY ' ? ' N^1.5 s' : ' proxy', DIM]]);
    const trX = W - pad, trY = Math.max(pad, opt.topY || 0) + 6;
    if (block(tr, trX, trY, 'tr') && ws > 0.05) {   // wind arrow: where the wind blows, relative to the nose (up = from behind)
      font(0.95); const lh = Math.round(fs * 1.38), wy = trY + lh * (fin(hdE) || fin(hdT) ? 1.5 : 0.5);
      const to = 90 - Math.atan2(w[1], w[0]) / D2R, rel = (to - hdgT) * D2R, ax = taken[taken.length - 1][0] - 8, L = 6;
      stroke(() => { ctx.moveTo(ax - L * Math.sin(rel), wy + L * Math.cos(rel)); ctx.lineTo(ax + L * Math.sin(rel), wy - L * Math.cos(rel)); ctx.lineTo(ax + L * Math.sin(rel) - 4 * Math.sin(rel - 0.5), wy - L * Math.cos(rel) + 4 * Math.cos(rel - 0.5)); ctx.moveTo(ax + L * Math.sin(rel), wy - L * Math.cos(rel)); ctx.lineTo(ax + L * Math.sin(rel) - 4 * Math.sin(rel + 0.5), wy - L * Math.cos(rel) + 4 * Math.cos(rel + 0.5)); }, gust, 1.5);
    }

    const br = [];
    if (fin(f.familiarity)) br.push([['FAM ', DIM], [f.familiarity.toFixed(2), brain]]);
    if (f.arbiter && fin(f.arbiter.commit)) br.push([['COMMIT ', DIM], [f.arbiter.commit.toFixed(2), brain]]);
    if (fin(hu.heading_conf)) br.push([['HDG CONF ', DIM], [hu.heading_conf.toFixed(2), brain]]);
    if (fin(hu.home_familiarity)) br.push([['HOME FAM ', DIM], [hu.home_familiarity.toFixed(2), brain]]);
    if (fin(hu.search_alt_m)) br.push([['SEARCH ', DIM], [hu.search_alt_m.toFixed(1) + ' m', brain], [' · size ' + (fin(hu.size_est_m) ? hu.size_est_m.toFixed(1) : '?') + ' m' + (hu.inspecting ? ' · inspecting' : ''), DIM]]);
    const hm = f.home;
    if (hu.landing_stage && !(hm && 'stage' in hm)) br.push([['LAND ', DIM], [String(hu.landing_stage).toUpperCase(), brain]]);
    if (hm && typeof hm === 'object' && 'stage' in hm && (!f.hud || hu.landing_stage)) {
      const st = hu.landing_stage ? String(hu.landing_stage).toUpperCase() : hm.touchdown ? 'TOUCHDOWN' : ['CRUISE', 'FINAL', 'DESCENT'][hm.stage] || String(hm.stage);
      br.push([['LAND ', DIM], [st, brain], [(hm.searching ? ' · searching' : '') + (hm.pad_visible ? ' · pad seen' : '') + (hm.locked ? ' · locked' : ''), DIM]]);
    }
    const bY = Math.min(H, opt.bottomY || H) - pad;
    // motor bars (1 front-right, then clockwise), to the left of the text block
    const ms = q.motors;
    const bw = 5 * s, bh = 26 * s, mw = 4 * bw + 3 * 3 + 2;
    if (br.length) block(br, W - pad, bY, 'br');
    if (ms && ms.length === 4) {
      const lastBr = taken[taken.length - 1], mx = (br.length && lastBr ? lastBr[0] - 12 : W - pad) - mw, my = bY - bh - 12 * s;
      if (claim([mx - 2, my - 2, mw + 4, bh + 16 * s])) {
        ms.forEach((m, i) => { const x = mx + i * (bw + 3), hgt = clamp(m, 0, 1) * bh; stroke(() => { ctx.rect(x, my, bw, bh); }, DIM, 0.8); ctx.fillStyle = TRUTH; ctx.fillRect(x + 1, my + bh - hgt, bw - 2, hgt); });
        font(0.72); text('MOT', mx + mw / 2, my + bh + 7 * s, DIM, 'center');
      }
    }
  }
  return { draw, set on(v) { on = !!v; if (!on) { size(); ctx.clearRect(0, 0, W, H); } }, get on() { return on; } };
};
