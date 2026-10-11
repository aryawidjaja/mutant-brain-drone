// Mutant brain drone: the workbench. Three modes in one page: Simulator (live, app/worker.js), Results (recorded runs
// and summary charts), Paper (guide.js into #guide). Classic script so Results and Paper open from file://; three.js is
// loaded with import() through the import map, as in flight-test-the-fly. Drawing helpers: panels.js, scene3d.js.
//
// Data written by the export scripts (each optional; the app shows a labelled mock or banner without it):
//   data/parity.js         window.PARITY = {pass: bool, quad: {...}, brain: {...}}      (hides the "not verified" banner)
//   data/summary.json|.js  window.APP_SUMMARY = {stubbornness_frontier, h1_success_vs_severity, recog_auroc, _source...}
//   runs/manifest.json     {runs: [{file, label}]}; runs/<file> = {meta: {scenario, arbiter, seed, git, sim_version, label},
//                          world: <C5>, frames: [<C6 frame>, ...] at 10 Hz; frame.eye optional, {w, h, b64} or {w, h, rgb}}
//   runs/bundle.js         window.RUN_BUNDLE = {manifest, runs: {file: run}}   (for file://)
//   data/tuned_params.js   window.TUNED = {_source, <arm>: params}; window.YAW_SURROGATE = {K, tau_s, wc_radps, ...}
//   sim/vehicle_params.js  ES module; SOURCES = {name: {value, unit, source}} (P/default also accepted)
'use strict';
(async () => {
const $ = (id) => document.getElementById(id);
const P = window.Panels, D2R = Math.PI / 180;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const HTTP = location.protocol !== 'file:';
let css = {};
const readCss = () => { const s = getComputedStyle(document.documentElement); for (const k of ['bg', 'panel', 'panel2', 'ink', 'muted', 'line', 'line2', 'a', 'b', 'gust', 'sky', 'chip', 'flag', 'mono']) css[k] = s.getPropertyValue('--' + k).trim(); css.dobj = s.getPropertyValue('--d-obj').trim(); css.dhome = s.getPropertyValue('--d-home').trim(); css.dsearch = s.getPropertyValue('--d-search').trim(); };
readCss();
const msg = (id, html) => { const el = $(id); el.innerHTML = html || ''; el.classList.toggle('on', !!html); };
const showErr = (e) => { console.error(e); msg('err', esc(String(e && e.message || e))); };

// ---------------------------------------------------------------- three.js (optional: the rest works without it)
let S3 = null;
try {
  const THREE = await import('three');
  let OC = null; try { OC = (await import('three/addons/controls/OrbitControls.js')).OrbitControls; } catch (e) { console.warn('OrbitControls failed to load', e); }
  const X = {}; // render-only addons; the scene falls back to simpler parts without any of them
  await Promise.all([['objects/Sky.js', (m) => { X.Sky = m.Sky; }], ['loaders/OBJLoader.js', (m) => { X.OBJLoader = m.OBJLoader; }],
    ['utils/BufferGeometryUtils.js', (m) => { X.mergeGeometries = m.mergeGeometries; X.mergeVertices = m.mergeVertices; }]]
    .map(([f, set]) => import('three/addons/' + f).then(set, (e) => console.warn(f + ' failed to load', e))));
  S3 = window.makeScene3D(THREE, OC, $('view'), X);
} catch (e) { console.warn('three.js failed to load', e); const d = document.createElement('div'); d.className = 'nosim on'; d.textContent = '3D view unavailable (three.js could not load). The map and panels still work.'; $('view').prepend(d); }

// ---------------------------------------------------------------- knobs -> C1 scenario
const ARBS = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine'];
const KN = [ // [group, id, label, type, default, a, b, step, tooltip]
  ['Mission', 'seed', 'Seed', 'num', 0, 0, 99999, 1, 'World layout, gusts and noise all follow this seed. The paper reports test seeds 1000–1299.'],
  ['Mission', 'tkind', 'Target', 'sel', 'tent', ['tent', 'backpack'], 0, 0, 'Kind of object the drone is shown and must find'],
  ['Mission', 'gal', 'Picture', 'gallery'],
  ['Mission', 'dist', 'Distance', 'rng', 40, 10, 120, 1, 'Mission distance: home to target, metres', (v) => v + ' m'],
  ['Mission', 'tbear', 'Bearing', 'rng', 55, 0, 359, 1, 'Compass bearing of the target from home (0 = north, clockwise)', (v) => v + '°'],
  ['Mission', 'tyaw', 'Target yaw', 'rng', 0, 0, 359, 1, 'Rotation of the target object', (v) => v + '°'],
  ['Mission', 'hx', 'Home x', 'num', 0, -100, 100, 1, 'Home position, metres east'],
  ['Mission', 'hy', 'Home y', 'num', 0, -100, 100, 1, 'Home position, metres north'],
  ['Mission', 'maxt', 'Time limit', 'rng', 300, 60, 600, 30, 'Mission time limit', (v) => v + ' s'],
  ['World', 'land', 'Landscape', 'sel', 'meadow', ['meadow', 'forest', 'urban'], 0, 0, 'Kind of world: open meadow, forest, or urban blocks'],
  ['World', 'dens', 'Obstacles', 'rng', 0.3, 0, 1, 0.05, 'Obstacle density (0 = none, 1 = densest)', (v) => v.toFixed(2)],
  ['World', 'light', 'Light', 'rng', 1, 0.1, 1, 0.05, 'Light level (1 = full daylight)', (v) => v.toFixed(2)],
  ['World', 'fogon', 'Fog', 'chk', false, 0, 0, 0, 'Fog on or off'],
  ['World', 'fog', 'Visibility', 'rng', 30, 5, 100, 1, 'Fog visibility distance', (v) => v + ' m', 'fogon'],
  ['Wind', 'wsp', 'Mean wind', 'rng', 1, 0, 8, 0.1, 'Mean wind speed', (v) => v.toFixed(1) + ' m/s'],
  ['Wind', 'wdir', 'From', 'rng', 270, 0, 359, 1, 'Direction the wind blows FROM (compass, 0 = north)', (v) => v + '°'],
  ['Wind', 'turb', 'Gusts', 'sel', 'light', ['none', 'light', 'moderate', 'severe'], 0, 0, 'Dryden turbulence intensity (MIL-F-8785C, low altitude)'],
  ['Faults', 'decoy', 'Decoy', 'sel', 'none', ['none', 'swap', 'beside', 'route'], 0, 0, 'none; swap = the look-alike stands on the reported spot and the real target sits 10 m to the side (the swap control; the study draws 8–14 m in a random direction); beside = 6 m from the target; route = halfway along the way'],
  ['Faults', 'dkind', 'Decoy kind', 'sel', 'tent', ['tent', 'backpack'], 0, 0, 'Kind of the decoy object', null, 'decoy'],
  ['Faults', 'dsim', 'Similarity', 'rng', 0.8, 0.5, 0.95, 0.05, 'How much the decoy looks like the target', (v) => v.toFixed(2), 'decoy'],
  ['Faults', 'dt', 'Decoy at', 'rng', 0, 0, 120, 1, 'Time the decoy appears', (v) => v + ' s', 'decoy'],
  ['Faults', 'occ', 'Occlusion', 'chk', false, 0, 0, 0, 'Hide the target for a while'],
  ['Faults', 'occt', 'Starts', 'rng', 0, 0, 20, 1, 'Seconds after the target is first seen within 10 m (the study uses 0)', (v) => '+' + v + ' s', 'occ'],
  ['Faults', 'occd', 'Lasts', 'rng', 5, 1, 10, 0.5, 'Occlusion duration', (v) => v + ' s', 'occ'],
  ['Faults', 'comp', 'Compass ½', 'chk', false, 0, 0, 0, 'Half-silence the compass (polarisation sensors on one side)'],
  ['Faults', 'compt', 'Starts', 'rng', 20, 0, 200, 1, 'Compass fault start', (v) => v + ' s', 'comp'],
  ['Faults', 'eyec', 'Dead eye cols', 'chk', false, 0, 0, 0, 'Kill a fraction of the eye columns'],
  ['Faults', 'eyect', 'Starts', 'rng', 20, 0, 200, 1, 'Eye fault start', (v) => v + ' s', 'eyec'],
  ['Faults', 'eyecf', 'Fraction', 'rng', 0.25, 0.1, 0.4, 0.05, 'Fraction of dead columns', (v) => Math.round(v * 100) + '%', 'eyec'],
  ['Faults', 'eyecs', 'First col', 'rng', 27, 0, 71, 1, 'First dead column of 72 (5° each, left to right across the panorama: column 0 looks straight back, columns 35–36 straight ahead)', (v) => String(v), 'eyec'],
  ['Faults', 'sun', 'False sun', 'chk', false, 0, 0, 0, 'A false sun (bright reflection) shifts the compass'],
  ['Faults', 'sunt', 'Starts', 'rng', 20, 0, 200, 1, 'False sun start', (v) => v + ' s', 'sun'],
  ['Faults', 'suno', 'Offset', 'rng', 60, 30, 120, 5, 'False sun offset', (v) => v + '°', 'sun'],
  ['Faults', 'spd', 'Speed error', 'chk', false, 0, 0, 0, 'Miscalibrated speed (odometry scale)'],
  ['Faults', 'spdt', 'Starts', 'rng', 0, 0, 200, 1, 'Speed error start', (v) => v + ' s', 'spd'],
  ['Faults', 'spds', 'Scale', 'rng', 1.2, 0.7, 1.3, 0.05, 'Odometry scale factor', (v) => '×' + v.toFixed(2), 'spd'],
  ['Brain', 'arb', 'Arbiter', 'sel', 'ring', ARBS, 0, 0, 'Which arbiter steers: ring hub (under test), ring without recurrence (control), or a rival'],
  ['Brain', 'prm', 'Params', 'sel', 'tuned', ['tuned', 'default'], 0, 0, 'Arbiter parameters: the frozen tuned values (file named below) or the code defaults'],
  ['Brain', 'gain', 'Ring gain', 'rng', 1, 0, 3, 0.01, 'Recurrent gain of the ring (0 = no memory; same as ring_static). Selecting the ring with tuned params sets the tuned value.', (v) => v.toFixed(2), 'arb=ring'],
  ['Brain', 'lf', 'Learning flight', 'chk', true, 0, 0, 0, 'Bee-style learning flight: the target is memorised from 5 views per second along a zig-zag arc 2–7 m out. Off = one photo from 4 m.'],
  ['Brain', 'lfs', 'Duration', 'rng', 10, 1, 30, 1, 'Learning flight length (the study uses 10 s)', (v) => v + ' s', 'lf'],
  ['Brain', 'ver', 'Version', 'sel', '1', ['1', '2'], 0, 0, 'v1 = the brain of the published first study; v2 = the robust brain under test'],
  ['Brain', 'hd', 'Heading', 'sel', 'hring', ['hring', 'kf', 'cf', 'v1c'], 0, 0, 'How the v2 brain keeps its heading', null, 'ver=2'],
  ['Brain', 'mhome', 'Home snapshot', 'chk', true, 0, 0, 0, 'v2: remember the home pad by sight', null, 'ver=2'],
  ['Brain', 'mland', 'Landing', 'chk', true, 0, 0, 0, 'v2: land on the pad', null, 'ver=2'],
  ['Brain', 'mfront', 'Insect front end', 'chk', true, 0, 0, 0, 'v2: colour-opponent, oriented front end for the memory', null, 'ver=2'],
  ['Brain', 'minsp', 'Inspection rule', 'chk', true, 0, 0, 0, 'v2: look from two sides before stopping', null, 'ver=2'],
  ['Brain', 'mgate', 'Evidence-gated ring', 'chk', true, 0, 0, 0, 'v2: arbiters only listen to objects they can see', null, 'ver=2'],
  ['Brain', 'tip', 'Operator tip', 'sel', 'default', ['default', 'poor', 'none'], 0, 0, 'How good the rough direction given to the drone is', null, 'ver=2'],
  ['Brain', 'yaw', 'Yaw surrogate', 'chk', false, 0, 0, 0, 'Exploratory, live view only: adds a yaw-rate damping term from a transfer function fitted to the Shiu-model fly yaw circuit. Off in every reported run.'],
];
const OPTNAME = { ver: { 1: 'v1 (published study)', 2: 'v2 (robust, under test)' }, hd: { hring: 'Heading ring', kf: 'Kalman filter', cf: 'Complementary filter', v1c: 'v1 sun compass' },
  tip: { default: 'Default', poor: 'Poor', none: 'None (search from scratch)' } };
const MODK = [['mhome', 'home'], ['mland', 'landing'], ['mfront', 'front'], ['minsp', 'inspect'], ['mgate', 'gated_arbiter']];
const HDEST = { hring: 'ring', kf: 'kf', cf: 'cf' };
const TIP_POOR = { bearing_sd_deg: 30, dist_sd: 0.3 }; // this app's "poor" tip (the default is 8 deg, 10 %)
// which v2 scenario fields app/sim/scenario.js accepts (read from its schema; refreshed while the page is open)
const SUPPORT = { version: false, modules: false, tip: false };
const ARBNAME = { ring: 'Ring hub', ring_static: 'Ring, no recurrence', vote: 'Static vote', sum: 'Vector sum', accumulator: 'Leaky accumulator', state_machine: 'State machine' };
const K = Object.fromEntries(KN.filter((k) => k[3] !== 'gallery').map((k) => [k[1], k[4]]));
K.pic = null; // target picture: gallery index 0-3 for the target kind, or null = drawn from the seed
const TUNED = window.TUNED || null, YAWFIT = window.YAW_SURROGATE || null, YAW_K = 0.013; // rad/s per Hz: about half the yaw rate fed back at low frequency
if (!TUNED) K.prm = 'default';
const tunedGain = () => (TUNED && TUNED.ring && Number.isFinite(TUNED.ring.ring_gain) ? TUNED.ring.ring_gain : 1);
if (TUNED) K.gain = tunedGain();
function armParams() { // brain.params for the chosen arbiter; ring_gain travels in brain.ring_gain
  if (K.prm !== 'tuned' || !TUNED || !TUNED[K.arb]) return {};
  const p = { ...TUNED[K.arb] }; delete p.ring_gain; return p;
}
const r1 = (v) => Math.round(v * 100) / 100;
function scenario() {
  const home = [K.hx, K.hy, 0], b = (90 - K.tbear) * D2R, tp = [r1(home[0] + K.dist * Math.cos(b)), r1(home[1] + K.dist * Math.sin(b)), 0];
  // world spans [-W/2, W/2] x [-H/2, H/2] around the origin (app/sim/world.js): keep home and target 10 m inside it
  const size = [0, 1].map((i) => Math.max(80, Math.ceil((2 * Math.max(Math.abs(home[i]), Math.abs(tp[i])) + 20) / 10) * 10));
  const decoys = [];
  if (K.decoy !== 'none') {
    const pos = K.decoy === 'swap' ? tp.slice() : K.decoy === 'beside' ? [r1(tp[0] - 6 * Math.sin(b)), r1(tp[1] + 6 * Math.cos(b)), 0] : [r1((home[0] + tp[0]) / 2 - 3 * Math.sin(b)), r1((home[1] + tp[1]) / 2 + 3 * Math.cos(b)), 0];
    decoys.push({ kind: K.dkind, pos, swap: K.decoy === 'swap', similarity: K.dsim, t_s: K.dt });
    if (K.decoy === 'swap') { tp[0] = r1(tp[0] - 10 * Math.sin(b)); tp[1] = r1(tp[1] + 10 * Math.cos(b)); } // as scripts/scenario.py: the target moves off its spot (8-14 m there, 10 m here)
  }
  const faults = [];
  if (K.occ) faults.push({ kind: 'occlusion', t_s: K.occt, params: { duration_s: K.occd } });
  if (K.comp) faults.push({ kind: 'compass_half', t_s: K.compt, params: {} });
  if (K.eyec) faults.push({ kind: 'eye_columns', t_s: K.eyect, params: { frac: K.eyecf, start_col: K.eyecs } });
  if (K.sun) faults.push({ kind: 'false_sun', t_s: K.sunt, params: { offset_deg: K.suno } });
  if (K.spd) faults.push({ kind: 'speed_scale', t_s: K.spdt, params: { scale: K.spds } });
  const wf = K.wdir * D2R;
  return {
    seed: K.seed, world: { size_m: size, landscape: K.land, obstacle_density: K.dens, light: K.light, fog_m: K.fogon ? K.fog : null },
    home, target: { kind: K.tkind, pos: tp, yaw_deg: K.tyaw, ...(K.pic !== null ? { picture: K.pic } : {}) }, decoys,
    wind: { mean_mps: [r1(-K.wsp * Math.sin(wf)), r1(-K.wsp * Math.cos(wf)), 0], turbulence: K.turb }, faults,
    brain: { arbiter: K.arb, ring_gain: K.arb === 'ring' ? K.gain : 0, learning_flight_s: K.lf ? K.lfs : 0, params: armParams(), ...v2Brain() },
    mission: { max_time_s: K.maxt, dt_s: 0.002 },
  };
}
function v2Brain() { // brain.version / modules / module_params / tip, only as far as app/sim/scenario.js accepts them
  if (K.ver !== '2' || !SUPPORT.version) return {};
  const o = { version: 2 };
  if (SUPPORT.modules) {
    o.modules = [...(K.hd !== 'v1c' ? ['heading'] : []), ...MODK.filter(([k]) => K[k]).map(([, m]) => m)];
    if (K.hd !== 'v1c' && HDEST[K.hd] !== 'ring') o.module_params = { heading_estimator: HDEST[K.hd] };
  }
  if (SUPPORT.tip && K.tip !== 'default') o.tip = K.tip === 'none' ? null : { ...TIP_POOR }; // absent = default tip
  return o;
}
function fromScenario(sc) { // inverse of scenario(), for #s= links
  const sw = (sc.decoys || []).find((d) => d.swap), h = sc.home || [0, 0, 0], t = sw ? sw.pos : sc.target.pos, dx = t[0] - h[0], dy = t[1] - h[1];
  Object.assign(K, { seed: sc.seed | 0, hx: h[0], hy: h[1], tkind: sc.target.kind, tyaw: Math.round(sc.target.yaw_deg || 0), dist: Math.round(Math.hypot(dx, dy)),
    tbear: Math.round(((90 - Math.atan2(dy, dx) / D2R) % 360 + 360) % 360), maxt: sc.mission.max_time_s,
    land: sc.world.landscape, dens: sc.world.obstacle_density, light: sc.world.light ?? 1, fogon: sc.world.fog_m != null, fog: sc.world.fog_m || K.fog,
    turb: sc.wind.turbulence, arb: sc.brain.arbiter, gain: sc.brain.ring_gain ?? 1, lf: (sc.brain.learning_flight_s ?? 10) > 0,
    pic: Number.isInteger(sc.target.picture) ? sc.target.picture : null, prm: TUNED && Object.keys(sc.brain.params || {}).length ? 'tuned' : 'default' });
  if (K.lf) K.lfs = sc.brain.learning_flight_s ?? 10;
  const w = sc.wind.mean_mps; K.wsp = r1(Math.hypot(w[0], w[1])); if (K.wsp) K.wdir = Math.round(((Math.atan2(-w[0], -w[1]) / D2R) % 360 + 360) % 360);
  const br = sc.brain || {};
  K.ver = br.version === 2 ? '2' : '1';
  if (br.version === 2) {
    const ms = br.modules || ['heading', ...MODK.map(([, m]) => m)];
    MODK.forEach(([k, m]) => { K[k] = ms.includes(m); });
    K.hd = !ms.includes('heading') ? 'v1c' : ({ ring: 'hring', kf: 'kf', cf: 'cf' }[(br.module_params || {}).heading_estimator || 'ring'] || 'hring');
    K.tip = !('tip' in br) ? 'default' : br.tip === null ? 'none' : 'poor';
  }
  if (K.tkind === 'pad') K.tkind = 'tent'; // the pad is no longer a target choice (it remains the home landing pad)
  const d = (sc.decoys || [])[0];
  K.decoy = !d ? 'none' : d.swap ? 'swap' : Math.hypot(d.pos[0] - t[0], d.pos[1] - t[1]) < 8 ? 'beside' : 'route';
  if (d) Object.assign(K, { dkind: d.kind === 'pad' ? 'tent' : d.kind, dsim: d.similarity ?? 0.8, dt: d.t_s || 0 });
  const F = Object.fromEntries((sc.faults || []).map((f) => [f.kind, f]));
  K.occ = !!F.occlusion; if (F.occlusion) { K.occt = F.occlusion.t_s; K.occd = F.occlusion.params.duration_s ?? 5; }
  K.comp = !!F.compass_half; if (F.compass_half) K.compt = F.compass_half.t_s;
  K.eyec = !!F.eye_columns; if (F.eye_columns) { K.eyect = F.eye_columns.t_s; K.eyecf = F.eye_columns.params.frac ?? 0.25; K.eyecs = F.eye_columns.params.start_col ?? 0; }
  K.sun = !!F.false_sun; if (F.false_sun) { K.sunt = F.false_sun.t_s; K.suno = F.false_sun.params.offset_deg ?? 60; }
  K.spd = !!F.speed_scale; if (F.speed_scale) { K.spdt = F.speed_scale.t_s; K.spds = F.speed_scale.params.scale ?? 1.2; }
}
const b64u = { enc: (o) => btoa(unescape(encodeURIComponent(JSON.stringify(o)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
  dec: (s) => JSON.parse(decodeURIComponent(escape(atob(s.replace(/-/g, '+').replace(/_/g, '/'))))) };
const hashP = () => new URLSearchParams(location.hash.slice(1));
function writeHash() { const h = new URLSearchParams(); h.set('m', mode); if (mode === 'sim') { h.set('s', b64u.enc(scenario())); if (K.yaw) h.set('y', '1'); } history.replaceState(null, '', '#' + h.toString()); }
try { const s = hashP().get('s'); if (s) fromScenario(b64u.dec(s)); K.yaw = hashP().get('y') === '1'; } catch (e) { showErr(new Error('Could not read the scenario in the link: ' + e.message)); }

// knob DOM, built from KN
const kEls = {};
function buildKnobs() {
  const host = $('knobgroups'); let grp = null, box = null;
  for (const k of KN) {
    const [g, id, label, type, , a, b, step, tip, fmt, dep] = k;
    if (g !== grp) {
      grp = g; const d = document.createElement('details'); d.open = true;
      d.innerHTML = `<summary class="ph">${g}${helpBtn(HELP['g:' + g], g)}<span class="who" id="kw-${g}"></span></summary><div class="body"><div class="kn"></div></div>`; host.appendChild(d); box = d.querySelector('.kn');
    }
    if (g === 'Faults' && (type === 'chk' || id === 'decoy') && box.children.length) box.insertAdjacentHTML('beforeend', '<span class="sub"></span>');
    if (type === 'gallery') { box.insertAdjacentHTML('beforeend', `<label title="Show the drone a new target picture">${label}${helpBtn(HELP.gal, label)}</label><div class="gal wide" id="gal"></div>`); continue; }
    const hb = helpBtn(HELP[id] || tip, label), hid = 'hlp' + tipN;
    if (type === 'chk') {
      box.insertAdjacentHTML('beforeend', `<label class="ck" data-dep="${dep || ''}"><input type="checkbox" id="k-${id}"><span>${label}</span>${hb}</label>`);
    } else {
      const ctl = type === 'sel' ? `<select id="k-${id}" class="${type === 'sel' ? 'wide' : ''}">${a.map((o) => `<option value="${o}">${(OPTNAME[id] || {})[o] || ARBNAME[o] || o}</option>`).join('')}</select>`
        : type === 'num' ? `<input type="number" id="k-${id}" min="${a}" max="${b}" step="${step}">` : `<input type="range" id="k-${id}" min="${a}" max="${b}" step="${step}">`;
      box.insertAdjacentHTML('beforeend', `<label for="k-${id}" data-dep="${dep || ''}">${label}${hb}</label>${ctl}${type === 'sel' ? '' : `<output id="o-${id}" data-dep="${dep || ''}"></output>`}`);
    }
    const el = $('k-' + id); kEls[id] = { el, type, fmt, dep }; el.setAttribute('aria-describedby', hid);
    if (id === 'tip') box.insertAdjacentHTML('beforeend', '<p class="fine v2note" id="v2note"></p>');
    if (id === 'yaw') box.insertAdjacentHTML('beforeend', '<p class="fine" id="prmsrc" style="grid-column:1/-1;margin:2px 0 0"></p>');
    el.addEventListener(type === 'rng' ? 'input' : 'change', () => {
      K[id] = type === 'chk' ? el.checked : type === 'sel' ? el.value : +el.value;
      if (type === 'num') K[id] = clamp(Math.round(K[id] / step) * step || 0, a, b);
      if (id === 'tkind') K.pic = null;
      if ((id === 'arb' || id === 'prm') && K.arb === 'ring' && K.prm === 'tuned' && TUNED) K.gain = tunedGain();
      syncKnobs(); knobChanged();
    });
  }
}
function syncKnobs() {
  for (const [id, { el, type, fmt, dep }] of Object.entries(kEls)) {
    if (type === 'chk') el.checked = !!K[id]; else if (String(el.value) !== String(K[id])) el.value = K[id];
    const o = $('o-' + id); if (o) o.textContent = fmt ? fmt(K[id]) : type === 'num' ? '' : K[id];
    if (dep) {
      const off = dep.includes('=') ? K[dep.split('=')[0]] !== dep.split('=')[1] : dep === 'decoy' ? K.decoy === 'none' : !K[dep];
      el.disabled = off; el.classList.toggle('off', off); document.querySelectorAll(`[data-dep="${dep}"]`).forEach((x) => x.classList.toggle('off', off));
    }
  }
  const nf = ['occ', 'comp', 'eyec', 'sun', 'spd'].filter((k) => K[k]).length + (K.decoy !== 'none' ? 1 : 0);
  const kw = $('kw-Faults'); if (kw) kw.textContent = nf ? `${nf} on` : 'none';
  const kb = $('kw-Brain'); if (kb) kb.textContent = ARBNAME[K.arb];
  if (kEls.prm) { kEls.prm.el.disabled = !TUNED; }
  const ps = $('prmsrc');
  if (ps) ps.innerHTML = (K.prm === 'tuned' && TUNED ? `Params: tuned, from <span class="mono">${esc(TUNED._source)}</span>${/kt2/.test(TUNED._source) ? ' (2D kill-test tuning, a stand-in until the 3D tuning is frozen)' : ''}.` : 'Params: code defaults (untuned).')
    + (K.yaw ? ` Yaw surrogate on: exploratory, not used for any reported number${YAWFIT ? ` (fit error ${YAWFIT.rms_gain_err_db.toFixed(2)} dB, ${YAWFIT.rms_phase_err_deg.toFixed(1)}° rms)` : ''}.` : '');
  if (kEls.yaw) kEls.yaw.el.disabled = !YAWFIT;
  if (kEls.ver) { kEls.ver.el.disabled = !SUPPORT.version; if (!SUPPORT.version) kEls.ver.el.value = '1'; }
  const v2on = K.ver === '2' && SUPPORT.version;
  for (const id of ['hd', ...MODK.map(([k]) => k)]) if (kEls[id]) { const off = !v2on || !SUPPORT.modules; kEls[id].el.disabled = off; kEls[id].el.closest('label, select').classList.toggle('off', off); }
  if (kEls.tip) { const off = !v2on || !SUPPORT.tip; kEls.tip.el.disabled = off; kEls.tip.el.classList.toggle('off', off); }
  const vn = $('v2note');
  if (vn) vn.textContent = !SUPPORT.version ? 'Brain v2 controls: coming with v2 (this build of app/sim does not accept brain.version yet).'
    : !SUPPORT.tip ? 'Operator tip: coming with v2 (app/sim/scenario.js does not define brain.tip yet).' : '';
  document.querySelectorAll('#gal button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.i === 'seed' ? K.pic === null : b.dataset.kind === K.tkind && b.dataset.i === String(K.pic))));
}
let pendingReset = false;
function knobChanged() {
  writeHash();
  if (sim.running) { pendingReset = true; $('reset').textContent = 'Reset to apply'; $('reset').classList.add('pri'); }
  else resetSim();
}

// ---------------------------------------------------------------- target picture gallery
// The pictures are the appearance presets of app/sim/world.js (GALLERY): picking one sets scenario.target.picture, so
// the drone is shown, and the world contains, exactly that target. "Seed" leaves the choice to the world seed.
async function buildGallery() {
  const g = $('gal'); let G = null;
  if (HTTP) try { G = (await import(new URL('sim/world.js', location.href).href)).GALLERY; } catch (e) { console.warn('gallery: sim/world.js', e); }
  const add = (kind, i, title, ap) => {
    const b = document.createElement('button'); b.type = 'button'; b.dataset.kind = kind; b.dataset.i = String(i); b.title = title;
    const c = document.createElement('canvas'); c.width = c.height = 64; b.appendChild(c); P.pic(c, { kind, appearance: ap });
    b.onclick = () => { if (i === 'seed') K.pic = null; else { K.tkind = kind; K.pic = i; } syncKnobs(); knobChanged(); };
    g.appendChild(b);
  };
  if (G) {
    const s = document.createElement('button'); s.type = 'button'; s.dataset.i = 'seed'; s.textContent = 'seed'; s.title = 'Let the world seed choose the picture'; s.className = 'mono';
    s.onclick = () => { K.pic = null; syncKnobs(); knobChanged(); }; g.appendChild(s);
    for (const kind of ['tent', 'backpack']) G[kind].forEach(([base, accent, pattern], i) => add(kind, i, `${kind} ${i + 1}: ${pattern}`, { base, accent, pattern }));
  } else {
    for (const kind of ['tent', 'backpack']) add(kind, 'seed', kind + ' (picture drawn from the seed; open over http for the full gallery)',
      { base: { tent: [0.9, 0.45, 0.1], pad: [0.95, 0.85, 0.1], backpack: [0.85, 0.2, 0.1] }[kind], accent: [0.95, 0.9, 0.8], pattern: { tent: 'stripes', pad: 'chevron', backpack: 'stripes' }[kind] });
    g.querySelectorAll('button').forEach((b) => { b.onclick = () => { K.tkind = b.dataset.kind; K.pic = null; syncKnobs(); knobChanged(); }; });
  }
  syncKnobs();
}

// ---------------------------------------------------------------- shared view state (one "current frame" for both modes)
const view = { world: null, scen: null, frame: null, trail: { n: 0, xy: new Float32Array(2 * 60000), xyz: new Float32Array(3 * 60000) }, hist: { t: [], fam: [], commit: [] }, phases: [], dur: 300, T: 0 };
function resetView(world, scen, dur) {
  Object.assign(view, { world, scen, frame: null, dur, T: 0, phases: [] }); view.trail.n = 0; view.hist = { t: [], fam: [], commit: [] };
  if (S3) { S3.setWorld(world); S3.clearTrail(); }
  objTags();
  panelDirty = true;
}
// pi.home_est is kept in the drone's compass frame; rotate it by the compass error (true yaw - heading estimate) so
// the map, 3D view and home panel draw it in the world frame. The raw vector stays in pi.home_est_compass (hub dial).
function worldPi(f) {
  const pi = f && f.pi;
  if (!pi || pi.home_est_compass || !Number.isFinite(pi.heading_est_deg)) return f;
  const a = (f.quad.euler_deg[2] - pi.heading_est_deg) * D2R, [x, y] = pi.home_est, c = Math.cos(a), s = Math.sin(a);
  f.pi = { ...pi, home_est_compass: pi.home_est, home_est: [c * x - s * y, s * x + c * y] };
  return f;
}
function pushFrame(f) {
  const tr = view.trail;
  if (tr.n < 60000) { const p = f.quad.pos; tr.xy[2 * tr.n] = p[0]; tr.xy[2 * tr.n + 1] = p[1]; tr.xyz.set(p, 3 * tr.n); tr.n++; }
  view.hist.t.push(f.t); view.hist.fam.push(f.familiarity ?? 0); view.hist.commit.push(f.arbiter ? f.arbiter.commit ?? 0 : 0);
  if (!view.phases.length || view.phases[view.phases.length - 1][1] !== f.phase) view.phases.push([f.t, f.phase]);
}
// presence over time of world objects (from the scenario: a decoy appears at its t_s; the target is always there)
function present(o) {
  const sc = view.scen, t = view.frame ? view.frame.t : 0; if (!sc) return true;
  const ds = sc.decoys || [];
  if (Number.isFinite(o.t_s)) return t >= o.t_s;
  if (o.role === 'decoy') { const d = ds.find((d) => d.kind === o.kind && Math.hypot(d.pos[0] - o.x, d.pos[1] - o.y) < 0.5); return !d || t >= (d.t_s || 0); }
  return true;
}
const arbOf = () => (view.scen ? view.scen.brain.arbiter : K.arb);
const roleColor = () => P.roleCol(css, arbOf());

// ---------------------------------------------------------------- live simulator (worker)
const sim = { w: null, running: false, src: '', rtf: 0, cpu: 0, ok: HTTP, done: false, note: '' };
function noSim(why) {
  sim.ok = false;
  const el = $('nosim'); el.classList.add('on');
  el.innerHTML = `<div>The live simulator runs in a Web Worker, which browsers only allow over http.<br>Serve the folder with <code>npx serve app</code> and open the address it prints.<br><span style="font-size:11.5px">${esc(why || '')}</span><br><br>The Results and Paper modes work from here.</div>`;
}
const simOpts = () => (K.yaw && YAWFIT ? { yawSurrogate: { fit: { K: YAWFIT.K, tau_s: YAWFIT.tau_s, wc_radps: YAWFIT.wc_radps }, k: YAW_K } } : {});
function resetSim() {
  pendingReset = false; $('reset').textContent = 'Reset'; $('reset').classList.remove('pri');
  if (!sim.ok) { const sc = scenario(); resetView(null, sc, sc.mission.max_time_s); return; }
  if (sim.w) sim.w.terminate();
  let w; try { w = new Worker('worker.js', { type: 'module' }); } catch (e) { noSim(e.message); return; }
  sim.w = w; sim.done = false; setRunning(false);
  const sc = scenario();
  w.onmessage = (e) => { if (w === sim.w) onSim(e.data, sc); };
  w.onerror = (e) => { if (w === sim.w) showErr(new Error('Simulator worker: ' + (e.message || 'failed to start'))); };
  w.postMessage({ cmd: 'init', scenario: sc, speed: +$('speed').value, opts: simOpts() });
  resetView(null, sc, sc.mission.max_time_s);
}
function onSim(d, sc) {
  if (d.type === 'ready') {
    sim.src = d.source; sim.note = d.note || ''; sim.worldSrc = d.worldSrc || '';
    resetView(d.world, sc, sc.mission.max_time_s);
    if (sim.running) sim.w.postMessage({ cmd: 'run' }); // Run pressed while the worker was still loading
    msg('err', '');
    badge(); banners();
  } else if (d.type === 'frame') {
    if (mode !== 'sim') return;
    worldPi(d.frame);
    if (view.frame && d.frame.t < view.frame.t) return;
    if (!view.frame || d.frame.t !== view.frame.t) pushFrame(d.frame);
    view.frame = d.frame; view.T = d.frame.t; sim.rtf = d.rtf; sim.cpu = d.cpu; panelDirty = true;
  } else if (d.type === 'done') { sim.done = true; setRunning(false); }
  else if (d.type === 'error') showErr(new Error(d.message));
}
function setRunning(on) {
  if (on && sim.done) { resetSim(); on = true; }
  sim.running = on && sim.ok;
  if (sim.w) sim.w.postMessage({ cmd: sim.running ? 'run' : 'pause' });
  playIcon(); $('run').textContent = sim.running ? 'Pause' : sim.done ? 'Run again' : 'Run';
}
function badge() {
  const b = $('badge'), mock = mode === 'sim' ? sim.src === 'mock' : results.mockRun;
  b.classList.toggle('on', mock); b.textContent = 'MOCK physics + brain' + (mode === 'sim' && sim.worldSrc === 'app/sim/world.js' ? ' · real world' : '');
  b.title = mode === 'sim' ? 'app/sim/mission.js is not available yet, so the live view uses app/data/mock_mission.js: a kinematic point drone and a toy brain, for building the interface only. ' + sim.note : 'This replay is a mock run made by app/data/mock_mission.js, not a result.';
}
// poll for the real modules while on the mock; switch at the next reset (immediately when idle)
if (HTTP) setInterval(async () => {
  if (sim.src !== 'mock' || mode !== 'sim') return;
  try { const r = await fetch('sim/mission.js', { method: 'HEAD', cache: 'no-store' }); if (r.ok && !sim.running && (!view.frame || view.frame.t < 0.05 || sim.done)) resetSim(); else if (r.ok) sim.note = 'Real modules found: press Reset to switch.'; } catch (e) { /* offline */ }
}, 5000);

// ---------------------------------------------------------------- Run 20 seeds (exploratory)
let batchW = null;
function runBatch() {
  const out = $('batchout'); out.hidden = false;
  if (batchW) { batchW.terminate(); batchW = null; $('batch').textContent = 'Run 20 seeds'; out.innerHTML += '<p class="fine">Cancelled.</p>'; return; }
  if (!sim.ok) { out.innerHTML = '<p class="fine">Needs http (npx serve app).</p>'; return; }
  const sc = scenario(), seeds = Array.from({ length: 20 }, (_, i) => sc.seed + i);
  batchW = new Worker('worker.js', { type: 'module' }); $('batch').textContent = 'Cancel';
  out.innerHTML = `<span class="ph">Exploratory · not part of the paper</span><p class="fine" id="bprog">Seeds ${seeds[0]}–${seeds[19]}: starting…</p>`;
  batchW.onmessage = (e) => {
    const d = e.data;
    if (d.type === 'progress') $('bprog').textContent = `Seed ${d.seed}: ${d.i} / ${d.n} done`;
    if (d.type === 'error') { out.innerHTML += `<p class="fine">${esc(d.message)}</p>`; batchW.terminate(); batchW = null; $('batch').textContent = 'Run 20 seeds'; }
    if (d.type !== 'batch') return;
    const R = d.results, n = R.length, pc = (k) => Math.round((100 * R.filter((r) => r[k]).length) / n) + '%';
    const ts = R.map((r) => r.time_s).filter(Number.isFinite).sort((a, b) => a - b), med = ts.length ? ts[ts.length >> 1].toFixed(0) : '–';
    out.innerHTML = `<span class="ph">Exploratory · not part of the paper</span>
      <table><tr><td>Arbiter</td><td>${esc(ARBNAME[sc.brain.arbiter])}${d.source === 'mock' ? ' <span class="flag">mock</span>' : ''}</td></tr>
      <tr><td>Seeds</td><td class="num">${seeds[0]}–${seeds[19]}</td></tr><tr><td>Found target</td><td class="num">${pc('success_target')}</td></tr>
      <tr><td>Got home</td><td class="num">${pc('success_home')}</td></tr><tr><td>Stopped at decoy</td><td class="num">${pc('wrong_object_stop')}</td></tr>
      <tr><td>Collisions (mean)</td><td class="num">${(R.reduce((s, r) => s + (r.collisions || 0), 0) / n).toFixed(1)}</td></tr><tr><td>Median time</td><td class="num">${med} <span class="u">s</span></td></tr></table>
      <p class="fine">20 seeds, no confidence intervals: a quick look only. The paper's numbers come from the Python reference on held-out test seeds.</p>`;
    batchW.terminate(); batchW = null; $('batch').textContent = 'Run 20 seeds';
  };
  batchW.postMessage({ cmd: 'batch', scenario: sc, seeds, opts: simOpts() });
}

// ---------------------------------------------------------------- Results mode
const results = { list: null, get: null, run: null, mockRun: false, playing: false, S: null, mockS: false };
async function loadRunSource() {
  const fromJSON = async (man, mock) => ({ list: man.runs, mock, get: async (f) => { const r = await fetch('runs/' + f); if (!r.ok) throw new Error(`Could not load runs/${f} (${r.status})`); return r.json(); } });
  if (HTTP) {
    for (const [f, mock] of [['runs/manifest.json', false], ['runs/mock_manifest.json', true]]) {
      try { const r = await fetch(f, { cache: 'no-cache' }); if (r.ok) return fromJSON(await r.json(), mock); } catch (e) { /* next */ }
    }
  }
  for (const [f, g, mock] of [['runs/bundle.js', 'RUN_BUNDLE', false], ['runs/mock_bundle.js', 'MOCK_RUN_BUNDLE', true]]) {
    await new Promise((ok) => { const s = document.createElement('script'); s.src = f; s.onload = ok; s.onerror = () => { s.remove(); ok(); }; document.head.appendChild(s); });
    if (window[g]) return { list: window[g].manifest.runs, mock, get: async (file) => window[g].runs[file] };
  }
  throw new Error('No recorded runs found (runs/manifest.json or runs/bundle.js).');
}
async function loadSummary() {
  if (HTTP) try { const r = await fetch('data/summary.json', { cache: 'no-cache' }); if (r.ok) return [await r.json(), false]; } catch (e) { /* next */ }
  if (window.APP_SUMMARY) return [window.APP_SUMMARY, false];
  if (window.MOCK_SUMMARY) return [window.MOCK_SUMMARY, true];
  return [null, false];
}
async function enterResults() {
  try {
    if (!results.list) {
      const src = await loadRunSource(); Object.assign(results, { list: src.list, get: src.get, mockRun: src.mock });
      $('runsel').innerHTML = src.list.map((r, i) => `<option value="${i}">${esc(r.label || r.file)}</option>`).join('');
      $('runsel').onchange = () => selectRun(+$('runsel').value);
      [results.S, results.mockS] = await loadSummary();
    }
    if (!results.run) await selectRun(0); else showRunFrame();
    drawCharts(); banners(); badge();
  } catch (e) { showErr(e); }
}
async function selectRun(i) {
  const entry = results.list[i], run = await results.get(entry.file);
  const fr = run.frames, dur = fr.length ? fr[fr.length - 1].t : 1;
  fr.forEach(worldPi);
  results.run = run; results.playing = false;
  resetView(run.world, run.meta.scenario || null, dur);
  for (const f of fr) pushFrame(f);
  view.T = 0; showRunFrame(); playIcon();
  const m = run.meta, o = fr.length ? fr[fr.length - 1].outcome || {} : {};
  const row = (k, v) => `<tr><td>${k}</td><td class="num">${esc(v)}</td></tr>`;
  $('runmeta').innerHTML = (m.label ? `<tr><td class="wrap" colspan="2" style="text-align:left">${esc(m.label)}</td></tr>` : '') + row('Arbiter', ARBNAME[m.arbiter] || m.arbiter) + row('Seed', m.seed) + row('Git', m.git) + row('Sim version', m.sim_version) +
    (m.params_source ? row('Params', m.params_source.split('/').pop()) : '') +
    ['success_target', 'success_home', 'wrong_object_stop', 'collisions', 'time_s', 'path_len_m', 'home_error_m', 'max_bank_deg', 't_hold_occlusion_s', 't_release_decoy_s'].filter((k) => k in o).map((k) => row(k.replace(/_/g, ' '), o[k] === null ? '–' : typeof o[k] === 'number' ? String(+o[k].toFixed(2)) : String(o[k]))).join('');
}
function showRunFrame() {
  const fr = results.run.frames; if (!fr.length) return;
  const dt = fr.length > 1 ? (fr[fr.length - 1].t - fr[0].t) / (fr.length - 1) : 0.1;
  const i = clamp(Math.floor((view.T - fr[0].t) / dt), 0, fr.length - 1), j = Math.min(fr.length - 1, i + 1), a = clamp((view.T - fr[i].t) / dt, 0, 1);
  const f = { ...fr[i], quad: { ...fr[i].quad, pos: fr[i].quad.pos.map((v, k) => v + a * (fr[j].quad.pos[k] - v)) } };
  if (!f.eye) for (let k = i; k >= 0; k--) if (fr[k].eye) { f.eye = fr[k].eye; break; }
  view.frame = f; view.trail.n = i + 1; panelDirty = true;
}
function drawCharts() {
  const S = results.S, wm = results.mockS ? '<div class="wm">MOCK DATA</div>' : '';
  const put = (id, s, what) => { $(id).innerHTML = (s || `<p class="fine">No ${what} in the summary yet.</p>`) + wm; };
  const el = $('ch1'), W = Math.max(260, el.clientWidth - 16), H = Math.max(220, Math.min(340, el.clientHeight - 12 || 260));
  put('ch1', P.chartFrontier(S, W, H), 'stubbornness frontier'); put('ch2', P.chartSeverity(S, W, H), 'success-vs-severity data'); put('ch3', P.chartAuroc(S, W, H), 'recognition AUROC');
}

// ---------------------------------------------------------------- banners, model panel
function banners() {
  const par = window.PARITY;
  const pass = par && par.pass === true;
  $('parity').classList.toggle('on', mode === 'sim' && !pass);
  if (par && !pass) $('parity').innerHTML = 'Parity check against the Python reference did not pass: the live simulator may differ from the numbers in the paper. Details: <span class="mono">' + esc(JSON.stringify({ quad: par.quad, brain: par.brain }).slice(0, 200)) + '</span>';
  const mk = mode === 'sim' ? (sim.src === 'mock' ? 'MOCK physics and brain: <span class="mono">app/sim/mission.js</span> is not available yet, so this view uses a kinematic stand-in (<span class="mono">app/data/mock_mission.js</span>) for building the interface. Nothing here is a result.' : '')
    : mode === 'results' ? [results.mockRun && 'Mock replay: the recorded run is synthetic (made by the mock simulator).', results.mockS && 'Mock summary: the charts show synthetic numbers (<span class="mono">data/mock_summary.js</span>), not results.'].filter(Boolean).join(' ') : '';
  msg('mock', mk);
}
const SIMP = [ // each line checked against the code it names (app/sim, scripts)
  ['Rigid body', '6-DOF quadrotor, RK4 at 500 Hz; mass and inertia from the MuJoCo Menagerie bitcraze_crazyflie_2 model (27 g; inertia from Landry 2015). Motor thrust, rotor-speed, torque and linear drag constants from Förster 2015, who measured a different unit (28.0 g, inertia about 16.6/16.7/29.3 × 10⁻⁶ kg m²): the mismatch is kept and stated.'],
  ['Motors', 'first-order lag, τ = 0.065 s, derived from Förster\'s discrete motor pole (0.9695 at 500 Hz); it lumps motor dynamics and some latency. Thrust saturates at the full-command value.'],
  ['Inner loop', 'a cascade PID (velocity → attitude → rate → mixer) stands in for the fly\'s halteres; the same for every arbiter.'],
  ['Wind', 'Dryden turbulence per MIL-F-8785C (low altitude), formulas as documented by MathWorks (Aerospace Blockset), discretised exactly per filter stage; gust axes along world x, y, z (not rotated into the mean wind), plus a mean wind.'],
  ['Eye', 'an analytic ray-cast 72 × 12 panorama (5° per pixel, 360° × 60°), not a rendered camera image. The mushroom body sees a separate 16 × 12 grey "look" of each visible object (an attention window that keeps the object filling the frame, blurred by acuity), rank-equalised. The recognition study (H4) uses MuJoCo renders instead.'],
  ['Visibility', 'an object is seen when present, large enough (≥ 4°), above fog contrast 0.2, with 2 of 3 sight lines clear of obstacles, not in a dead eye column and not occluded. A perfect tracker keeps each object in the same slot (slot order random per seed).'],
  ['Optic flow and looming', 'geometric optic flow from 72 horizontal ray-casts (range ≤ 20 m) plus noise, not a model of motion detectors (no flyvis, no LPLC2 or HS cells). Flow balancing and looming avoidance are added after the arbiter, at the descending-neuron level.'],
  ['Mushroom body', 'Ardin et al. 2016-style: 20,000 Kenyon cells, 10 random inputs each, 1% active (k-winners), one-shot binary depression of the output weights.'],
  ['Learning flight', 'a set of views, not a flown manoeuvre: 5 looks per second along a zig-zag arc 2–7 m from the target, stored before t = 0. Off = one look from 4 m.'],
  ['Path integrator', 'Stone et al. 2017, reimplemented from the paper\'s text, with three stated departures: a multiplicative CPU4 update (as the prose describes), rectified-linear TN2 speed cells, and a population-vector readout of CPU4 (no CPU1 steering layer). The home vector lives in the compass frame; compass faults rotate it.'],
  ['Ring hub', 'Sun, Yue & Mangan 2020-style, with departures: 32 neurons (not 8 per hemisphere), 30° excitation width, quasi-steady inhibition, activation clipped to [0, 1], state carried across ticks (that is its memory), tunable time constant; ring gain scales the whole recurrent loop.'],
  ['Stop rule', 'shared by every arbiter, outside it: stop within 2 m (horizontal) of a visible object whose 1-s mean evidence exceeds 0.6. Stopping at a look-alike counts as a wrong-object stop; then the home drive takes over and landing relies on the path integrator alone.'],
  ['Instruction', 'the drone is given a rough vector to the reported spot (bearing error 8°, distance error 10%, 1 SD), then searches an outward spiral around it.'],
];
async function modelPanel() {
  let params = null, why = '';
  if (HTTP) {
    try {
      if ((await fetch('sim/vehicle_params.js', { method: 'HEAD', cache: 'no-store' })).ok) { const m = await import(new URL('sim/vehicle_params.js', location.href).href); params = m.SOURCES || m.default || m.PARAMS || m.P || m.params || null; }
      else why = 'app/sim/vehicle_params.js is not written yet.';
    } catch (e) { why = 'Could not load app/sim/vehicle_params.js: ' + e.message; }
  } else why = 'Open over http to load app/sim/vehicle_params.js.';
  const t = $('model');
  if (params && typeof params === 'object') {
    const rows = Object.entries(params).filter(([k]) => !k.startsWith('_')).map(([k, v]) => {
      const o = v && typeof v === 'object' && 'value' in v ? v : { value: v };
      const val = Array.isArray(o.value) ? o.value.map((x) => +(+x).toPrecision(4)).join(', ') : typeof o.value === 'number' ? +o.value.toPrecision(5) : String(o.value);
      return `<tr><td>${esc(k)}</td><td class="num">${esc(val)} <span class="u">${esc(o.unit || '')}</span></td></tr>${o.source ? `<tr><td colspan="2" class="src">${esc(o.source)}</td></tr>` : ''}`;
    });
    t.innerHTML = `<thead><tr><th>Vehicle parameter</th><th>Value</th></tr></thead>` + rows.join('');
  } else t.innerHTML = `<tr><td class="wrap" colspan="2">Vehicle parameters with sources will be listed here. ${esc(why)} Mass and inertia come from the MuJoCo Menagerie <i>bitcraze_crazyflie_2</i> model; motor constants from a separately cited source.</td></tr>`;
  const yawLi = YAWFIT ? `<li><b>Yaw circuit (optional):</b> the author's earlier Shiu-model fly yaw circuit (flight-test-the-fly) is too slow for live use. The "Yaw surrogate" toggle uses a transfer function fitted to its measured frequency response (gain ${YAWFIT.K.toFixed(3)} Hz per °/s, delay ${(YAWFIT.tau_s * 1000).toFixed(1)} ms, corner ${YAWFIT.fc_hz.toFixed(1)} Hz; fit error ${YAWFIT.rms_gain_err_db.toFixed(2)} dB, ${YAWFIT.rms_phase_err_deg.toFixed(1)}° rms) as a small yaw-rate damping term. Exploratory, live view only, off by default and in every reported run.</li>` : '<li><b>Yaw circuit:</b> a fitted surrogate of the Shiu-model yaw circuit exists (app/sim/yaw_surrogate.js) but its fit file is not loaded here.</li>';
  const prmLi = `<li><b>Arbiter parameters:</b> ${TUNED ? `tuned values from <span class="mono">${esc(TUNED._source)}</span>${/kt2/.test(TUNED._source) ? ' (tuned in the 2D kill-test world, used here as a stand-in until the 3D tuning is frozen)' : ''}` : 'code defaults (no tuned file found)'}.</li>`;
  const gfxLi = '<li><b>Graphics:</b> Visual detail (grass, textures, foliage) is decorative; the drone\'s eye and collisions use the simplified world geometry. Credits: drone meshes from MuJoCo Menagerie <i>bitcraze_crazyflie_2</i> (MIT); ground, bark, rock, asphalt, concrete and fabric textures from ambientCG (CC0; Grass004, Ground037, Ground106, Bark014, Rock051, Asphalt031, Concrete034, Fabric061); sky from the three.js Sky addon (MIT, Preetham daylight model).</li>';
  $('simp').innerHTML = '<li class="ph" style="list-style:none;margin-left:-16px">Simplifications</li>' + SIMP.map(([a, b]) => `<li><b>${a}:</b> ${b}</li>`).join('') + yawLi + prmLi + gfxLi + (sim.src === 'mock' ? '<li><b>Now running:</b> the mock simulator, a kinematic point drone with a toy brain. None of the above applies until app/sim/mission.js loads.</li>' : '');
}

// ---------------------------------------------------------------- panels
let panelDirty = true, lastPanel = 0;
function objTags() {
  const w = view.world; ['tagT', 'tagD0', 'tagD1', 'tagD2'].forEach((id) => { $(id).hidden = true; });
  $('tagH').hidden = !w;
}
function placeTags() {
  const w = view.world; if (!w || !S3) return;
  const put = (id, x, y, z, txt) => { const el = $(id), p = S3.project(x, y, z); if (!p || p[0] < 0 || p[1] < 0 || p[0] > $('view').clientWidth || p[1] > $('view').clientHeight) { el.hidden = true; return; } el.hidden = false; if (txt) el.textContent = txt; el.style.transform = `translate(${Math.round(p[0] + 6)}px, ${Math.round(p[1] - 18)}px)`; };
  let d = 0;
  for (const o of w.objects) {
    const here = present(o), z = o.kind === 'tent' ? 1.6 : 0.7;
    if (o.role === 'target') { here ? put('tagT', o.x, o.y, z, `target · slot ${o.slot}`) : ($('tagT').hidden = true); }
    else if (d < 3) { const id = 'tagD' + d++; here ? put(id, o.x, o.y, z, `decoy · slot ${o.slot} · sim ${(o.appearance.similarity ?? 0).toFixed(2)}`) : ($(id).hidden = true); }
  }
  put('tagH', w.home[0], w.home[1], 0.3);
}
const n2 = (x) => (typeof x === 'number' ? x.toFixed(2) : String(x));
function vizRows(f) { // the arbiter's own state, readable
  const a = f.arbiter || {}, v = a.viz || {}, K = (f.objects || []).length;
  if (a.name === 'state_machine') return [['state', P.SM_STATES[v.state] ?? v.state], ['committed to', v.target >= 0 ? `slot ${v.target}` : 'none'],
    ['evidence', Array.from(v.evidence || []).map((x, k) => `${k}: ${n2(x)}`).join(' · ')], ['rejected', (v.rejected || []).map((x, k) => (x ? k : null)).filter((x) => x !== null).join(', ') || 'none']];
  if (a.name === 'accumulator') { const lab = (k) => (k < K ? `slot ${k}` : k === K ? 'home' : 'search'); return [['evidence', Array.from(v.evidence || []).map((x, k) => `${lab(k)} ${n2(x)}`).join(' · ')], ['choice', lab(v.choice)]]; }
  if (a.name === 'sum' && v.vectors) return Object.entries(v.vectors).map(([k, u]) => [k, `${Math.hypot(u[0], u[1]).toFixed(2)} at ${(Math.atan2(u[1], u[0]) / D2R).toFixed(0)}°`]);
  return Object.entries(v).map(([k, x]) => [k, Array.isArray(x) ? x.slice(0, 8).map(n2).join(' · ') : typeof x === 'object' ? JSON.stringify(x) : n2(x)]);
}
function drawPanels() {
  const f = view.frame;
  P.map($('map'), view.world, view.trail, f, css, { col: roleColor(), present });
  const tgt = view.world && view.world.objects.find((o) => o.role === 'target');
  P.pic($('pic'), tgt || { kind: K.tkind, appearance: {} });
  if (!f) return;
  P.eye($('eye'), f.eye);
  P.fam($('fam'), view.hist, f.t, css);
  P.hub($('hub'), f, css);
  P.homev($('homev'), f.pi, css);
  P.att($('att'), f, css);
  const arb = f.arbiter ? f.arbiter.name : arbOf();
  $('hubwho').textContent = ARBNAME[arb] || arb;
  $('hubcmdkey').innerHTML = `<i style="background:${roleColor()}"></i>heading command`;
  const FR = 'Compass frame (the drone\'s own north; true north only when the compass is right). Ink arrow = own heading, coloured arrow = heading command.';
  $('hubnote').textContent = (/^ring/.test(arb) ? 'Ring of 32 heading neurons: brightness = activity. Bars outside the ring = input from each drive at that heading.'
    : arb === 'vote' ? 'Static vote: the summed input bump (no dynamics, no memory); the command is its interpolated peak.'
      : arb === 'sum' ? 'Vector sum: one arrow per drive (teal objects, violet home, grey search); the command is their sum.'
        : arb === 'accumulator' ? 'Leaky accumulator: bar = evidence per slot (dashed = not visible), violet = home option; the largest wins (orange/blue).'
          : arb === 'state_machine' ? 'State machine: state name in the centre; bars = filtered evidence per slot, the committed target highlighted, × = rejected.' : '') + ' ' + FR;
  $('viz').innerHTML = vizRows(f).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${esc(v)}</td></tr>`).join('');
  $('famv').textContent = Number.isFinite(f.familiarity) ? `familiarity ${f.familiarity.toFixed(2)}` : '';
  if (f.pi) { const e = f.pi.home_est, t = f.pi.home_true; $('hverr').textContent = `error ${Math.hypot(e[0] - t[0], e[1] - t[1]).toFixed(1)} m`; }
  $('mindtxt').textContent = f.mind ? f.mind.text : '–'; $('mindtru').textContent = f.mind ? f.mind.truth : '–';
  const chips = (f.faults_active || []).map((k) => `<span class="flag warn">${esc(k.replace(/_/g, ' '))}</span>`);
  $('chips').innerHTML = `<span class="flag ph-on">${esc(f.phase)}</span>` + chips.join('') + (f.outcome ? `<span class="flag">${f.outcome.success_target ? 'found target' : f.outcome.wrong_object_stop ? 'stopped at decoy' : 'target not found'} · ${f.outcome.success_home ? 'home' : 'not home'}</span>` : '');
  const o = f.objects || [];
  const rows = [['Phase', f.phase], ['Commit', f.arbiter && Number.isFinite(f.arbiter.commit) ? f.arbiter.commit.toFixed(2) : '–'], ['Heading cmd', f.ring && Number.isFinite(f.ring.heading_cmd_deg) ? f.ring.heading_cmd_deg.toFixed(0) + '°' : '–'],
    ['Avoid yaw offset', f.avoid ? f.avoid.yaw_offset_deg.toFixed(0) + '°' : '–'], ['Looming', f.avoid ? f.avoid.loom.toFixed(2) : '–'], ['Altitude', f.quad.pos[2].toFixed(2) + ' m'],
    ['Body rates', f.quad.omega.map((v) => (v / D2R).toFixed(0)).join(', ') + ' °/s']];
  $('now').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td class="num">${esc(b)}</td></tr>`).join('') +
    `<tr class="sub"><td colspan="4" class="ph" style="padding-top:7px">Object slots</td></tr><tr><td>slot</td><td class="num">bearing</td><td class="num">strength</td><td class="num">seen</td></tr>` +
    o.map((s) => `<tr><td>${s.slot}</td><td class="num">${s.bearing_deg.toFixed(0)}°</td><td class="num">${s.strength.toFixed(2)}</td><td class="num">${s.visible ? 'yes' : '–'}</td></tr>`).join('');
  $('nowt').textContent = `t ${f.t.toFixed(1)} s`;
}

// ---------------------------------------------------------------- timeline
const axis = $('axis'), track = $('track');
const PHC = () => ({ learning: css.dsearch, search: css.line2, approach: css.dobj, home: css.dhome, done: css.ink });
function drawTimeline() {
  const dpr = Math.min(devicePixelRatio, 2), W = axis.clientWidth, H = axis.clientHeight;
  if (axis.width !== Math.round(W * dpr) || axis.height !== Math.round(H * dpr)) { axis.width = Math.round(W * dpr); axis.height = Math.round(H * dpr); }
  const c = axis.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const D = Math.max(1, view.dur), X = (t) => (t / D) * W, T = view.T, pc = PHC();
  c.fillStyle = css.line; c.fillRect(0, H - 9, W, 3);
  view.phases.forEach(([t0, ph], i) => { const t1 = i + 1 < view.phases.length ? view.phases[i + 1][0] : mode === 'sim' ? T : view.dur; c.fillStyle = pc[ph] || css.muted; c.fillRect(X(t0), H - 10, Math.max(1, X(Math.min(t1, mode === 'sim' ? T : t1)) - X(t0)), 5); });
  const step = [1, 2, 5, 10, 20, 30, 60, 120].find((s) => D / s <= W / 60) || 120;
  c.strokeStyle = css.muted; c.fillStyle = css.muted; c.font = `10.5px ${css.mono}`; c.textBaseline = 'top'; c.lineWidth = 1; c.beginPath();
  for (let t = 0; t <= D + 1e-9; t += step) { const x = Math.round(X(t)) + 0.5; c.moveTo(x, 0); c.lineTo(x, 6); c.textAlign = t === 0 ? 'left' : x > W - 20 ? 'right' : 'center'; c.fillText(t + (t === 0 ? ' s' : ''), t === 0 ? x + 2 : x, 8); }
  c.stroke();
  const x = Math.round(X(T)) + 0.5; c.strokeStyle = css.ink; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
  track.setAttribute('aria-valuemax', String(Math.round(D))); track.setAttribute('aria-valuenow', String(Math.round(T)));
  $('time').innerHTML = `${T.toFixed(1)} <small>/ ${D.toFixed(0)} s</small>`;
}
const PLAY = '<svg viewBox="0 0 16 16"><path d="M4.5 3v10l8-5z" fill="currentColor"/></svg>', PAUSE = '<svg viewBox="0 0 16 16"><path d="M4.5 3h2.2v10H4.5zM9.3 3h2.2v10H9.3z" fill="currentColor"/></svg>';
function playIcon() { const on = mode === 'sim' ? sim.running : results.playing; $('play').innerHTML = on ? PAUSE : PLAY; $('play').setAttribute('aria-label', on ? 'Pause' : 'Play'); }
function togglePlay() {
  if (mode === 'sim') setRunning(!sim.running);
  else if (mode === 'results' && results.run) { if (!results.playing && view.T >= view.dur) view.T = 0; results.playing = !results.playing; playIcon(); }
}
(() => { // scrub (Results only; the live sim cannot rewind)
  let drag = false;
  const to = (e) => { if (mode !== 'results' || !results.run) return; const r = track.getBoundingClientRect(); view.T = clamp(((e.clientX - r.left) / r.width) * view.dur, 0, view.dur); showRunFrame(); };
  track.addEventListener('pointerdown', (e) => { drag = true; track.setPointerCapture(e.pointerId); to(e); });
  track.addEventListener('pointermove', (e) => { if (drag) to(e); });
  track.addEventListener('pointerup', () => { drag = false; });
  track.addEventListener('keydown', (e) => { if (mode !== 'results' || !results.run) return; const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (d) { view.T = clamp(view.T + d * (e.shiftKey ? 10 : 1), 0, view.dur); showRunFrame(); e.preventDefault(); } });
})();

// ---------------------------------------------------------------- modes
let mode = 'sim';
function setMode(m) {
  mode = m; const wb = $('wb');
  document.body.classList.toggle('m-paper', m === 'paper'); document.body.classList.toggle('m-sim', m === 'sim');
  wb.classList.toggle('m-sim', m !== 'results'); wb.classList.toggle('m-results', m === 'results');
  document.querySelectorAll('#modes button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === m)));
  if (m === 'sim') { results.playing = false; resetSim(); }
  else { if (sim.running) setRunning(false); }
  if (m === 'results') enterResults();
  if (m === 'paper') window.scrollTo(0, 0);
  writeHash(); banners(); badge(); playIcon(); applyTheme(); requestAnimationFrame(resize);
}
document.querySelectorAll('[data-mode]').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });

// ---------------------------------------------------------------- flight display (HUD) over the 3D view
const HUD = window.makeHUD ? window.makeHUD($('hud'), $('view')) : null;
function setHud(onv, save) {
  if (!HUD) return; HUD.on = onv; $('hudbtn').setAttribute('aria-pressed', String(onv)); $('wb').classList.toggle('hud-on', onv);
  if (save) try { localStorage.setItem('mbd-hud', onv ? '1' : '0'); } catch (e) { /* private */ }
}
$('hudbtn').onclick = () => setHud(!(HUD && HUD.on), true);
const vRect = (el) => { if (!el || el.hidden || !el.offsetParent) return null; const a = el.getBoundingClientRect(), b = $('view').getBoundingClientRect(); return a.width && a.height ? [a.left - b.left, a.top - b.top, a.width, a.height] : null; };
let hudAvoid = [], hudAvoidT = 0, hudTop = 0, hudBot = 0;
function drawHud(now) {
  if (!HUD || !HUD.on) return;
  if (now - hudAvoidT > 250) { // overlays the display must not cover
    hudAvoidT = now; hudAvoid = ['badge', 'mapwrap', 'mind', 'nosim'].map((id) => vRect($(id))).filter(Boolean);
    const tools = document.querySelector('#view .tools'), tr = vRect(tools); if (tr) hudAvoid.push(tr);
    hudTop = tr ? tr[1] + tr[3] : 0; const mr = vRect($('mind')); hudBot = mr ? mr[1] : 0;
  }
  HUD.draw(view.frame, { avoid: hudAvoid, topY: hudTop, bottomY: hudBot, brain: roleColor(), flag: css.flag, gust: css.gust });
}

// ---------------------------------------------------------------- help: "?" buttons with one shared tooltip
const STUDY = 'Part of the study: the paper varies or fixes this.', EXPL = 'Exploratory: live view only, not part of any reported number.', VIEW = 'View only: changes nothing in the simulation.';
const HELP = {
  'g:Mission': 'Where the target is, what it looks like, and how long the drone has. ' + STUDY,
  'g:World': 'The landscape the drone flies through: obstacles, light and fog. ' + STUDY,
  'g:Wind': 'Steady wind plus random gusts that push the drone around. ' + STUDY,
  'g:Faults': 'Deliberate problems injected into the drone\'s senses or its world, to test how robust the brain is. ' + STUDY,
  'g:Brain': 'Which brain flies and which of its parts are switched on.',
  seed: 'Random seed. The same seed gives the same world layout, gusts and sensor noise, so a run can be repeated exactly. Any whole number; the paper\'s test seeds are 1000–1299 (v1) and 2000–2299 (v2). ' + STUDY,
  tkind: 'The object the drone is shown a picture of and must find: a tent or a backpack. ' + STUDY,
  gal: 'The picture shown to the drone before take-off (colours and pattern of the target). "seed" lets the world seed choose. ' + STUDY,
  dist: 'Straight-line distance from home to the target, in metres. The study uses 25–40 m; up to 120 m here. ' + STUDY,
  tbear: 'Compass direction of the target as seen from home (0° = north, 90° = east). ' + STUDY,
  tyaw: 'Which way the target object faces, in degrees. Matters for recognition: the memory must cope with new viewing angles. ' + STUDY,
  hx: 'Home position, metres east of the world centre. Home is where the drone takes off and must return to. ' + STUDY,
  hy: 'Home position, metres north of the world centre. ' + STUDY,
  maxt: 'Mission time limit in seconds. If the drone has not finished by then, the run ends as a failure. Typical 300 s. ' + STUDY,
  land: 'Kind of landscape: open meadow (few trees), forest (many trees) or urban blocks (buildings). Changes what the eye sees and what the drone must avoid. ' + STUDY,
  dens: 'How crowded the world is with trees, rocks or buildings, from 0 (empty) to 1 (densest). Typical 0.3. ' + STUDY,
  light: 'Overall brightness, from 0.1 (dusk) to 1 (full daylight). Dims what the eye sees, so recognition gets harder. ' + STUDY,
  fogon: 'Turn fog on or off. Fog washes out distant objects in the drone\'s eye. ' + STUDY,
  fog: 'Fog visibility: the distance in metres at which objects fade into the fog. Lower = thicker fog. Typical 20–60 m. ' + STUDY,
  wsp: 'Average wind speed in metres per second. The Crazyflie cruises at 2 m/s, so 2–3 m/s of wind is a lot for it. ' + STUDY,
  wdir: 'Direction the wind blows FROM, as a compass bearing (270° = from the west). ' + STUDY,
  turb: 'Random gusts on top of the steady wind (Dryden turbulence model, light to severe). Gusts shake the drone and its speed sensing. ' + STUDY,
  decoy: 'Add a look-alike object. Swap: the look-alike stands where the target was reported and the real one is moved aside (the hardest test). Beside: 6 m from the target. Route: halfway along the way. ' + STUDY,
  dkind: 'Kind of the look-alike object (tent or backpack). ' + STUDY,
  dsim: 'How much the look-alike resembles the target, from 0.5 (clearly different) to 0.95 (nearly identical). ' + STUDY,
  dt: 'Seconds after the start when the look-alike appears. ' + STUDY,
  occ: 'Hide the target for a while once the drone first sees it up close: does the brain keep heading for it? ' + STUDY,
  occt: 'Delay, in seconds after the target is first seen within 10 m, before it is hidden. The study uses 0. ' + STUDY,
  occd: 'How long the target stays hidden, in seconds. Typical 5 s. ' + STUDY,
  comp: 'Half of the drone\'s sky-compass sensors go silent, so its sense of direction gets worse. ' + STUDY,
  compt: 'Seconds after the start when the compass fault begins. ' + STUDY,
  eyec: 'A block of the eye\'s columns stops working (blind sector). ' + STUDY,
  eyect: 'Seconds after the start when part of the eye goes blind. ' + STUDY,
  eyecf: 'Fraction of the eye that goes blind, 10–40 %. ' + STUDY,
  eyecs: 'Where the blind sector starts: column 0 looks straight back, 35–36 straight ahead (72 columns, 5° each). ' + STUDY,
  sun: 'A false sun (for example a bright reflection) tricks the sky compass into a wrong direction. ' + STUDY,
  sunt: 'Seconds after the start when the false sun appears. ' + STUDY,
  suno: 'How far the false sun is from the real one, in degrees; the compass is off by this much. Typical 60°. ' + STUDY,
  spd: 'The drone\'s speed sense is miscalibrated, so it misjudges distances travelled and the way home. ' + STUDY,
  spdt: 'Seconds after the start when the speed error begins. ' + STUDY,
  spds: 'Speed scale error: ×1.2 means the drone thinks it flies 20 % faster than it does. ' + STUDY,
  arb: 'The decision hub that turns all urges (go to object, go home, search) into one heading. Ring hub = the insect-style ring attractor under test; the others are fair rivals and a no-memory control. ' + STUDY,
  prm: 'Arbiter settings: the frozen values tuned on training seeds (as in the paper) or the plain code defaults. ' + STUDY,
  gain: 'Strength of the ring\'s self-excitation, which is its short-term memory. 0 = no memory; higher = more stubborn (holds a hidden target longer but also chases a decoy longer). Typical 0.3–1.5. ' + STUDY,
  lf: 'Bee-style learning flight: before setting off, the drone memorises the target from many views (5 per second along a zig-zag arc). Off = a single photo. ' + STUDY,
  lfs: 'Length of the learning flight in seconds. The study uses 10 s. ' + STUDY,
  ver: 'Which brain flies. v1 = the brain of the published first study, unchanged. v2 = the more robust brain under test (better heading, home snapshot and landing, better recognition). ' + STUDY,
  hd: 'How the v2 brain keeps track of which way it faces. Heading ring = insect-style ring fed by the gyro and corrected by the sky compass; Kalman and complementary filters are engineering rivals; v1 sun compass = the old direct compass (heading module off). ' + STUDY,
  mhome: 'v2 home snapshot: the drone memorises how the home pad looks during take-off and steers by sight when it gets back, instead of trusting dead reckoning alone. ' + STUDY,
  mland: 'v2 landing: a final slow approach and a vertical descent onto the home pad. Off = the run ends when the drone thinks it is home. ' + STUDY,
  mfront: 'v2 insect front end: colour-opponent and orientation-sensitive pre-processing (like early insect vision) before the memory, instead of plain grey pixels. ' + STUDY,
  minsp: 'v2 inspection rule: before stopping at an object, look at it from two directions at least 20° apart, to catch look-alikes. ' + STUDY,
  mgate: 'v2 evidence gate: the hub only listens to objects that are currently visible, so it lets go of a look-alike faster. ' + STUDY,
  tip: 'The rough vector to the reported spot that the operator gives the drone before take-off. Default = bearing error about 8° and distance error 10 % (1 SD), as in the study; poor = 30° and 30 % (a choice of this app); none = no tip, the drone spirals out from home. ' + STUDY,
  yaw: 'Adds a small yaw-damping term from a transfer function fitted to the author\'s earlier fly yaw-circuit model. ' + EXPL,
  'p:eye': 'Eye and memory. Left: the low-resolution 360° picture the drone\'s memory sees (ahead in the middle). Right: the target picture it was shown. Below: familiarity, how strongly the current view matches the memory (0 to 1; the brain treats about 0.6 and above as a match), and commitment to an object. Brain state, recomputed live.',
  'p:hub': 'Steering hub. The ring of 32 heading neurons (or the rival arbiter) that combines the urges: objects (teal), home (violet), search (grey). The orange arrow is the heading the brain chooses; the light arrow is its current heading. Brain state, in the brain\'s own compass frame.',
  'p:homev': 'Home-vector memory. The path integrator\'s estimate of the way home (solid) next to the simulator\'s truth (faint). The gap between them is the navigation error, in metres.',
  'p:att': 'Attitude, motors and wind. Artificial horizon (bank and pitch), the four motor commands (0–1 of maximum thrust) and the wind at the drone. Simulator truth.',
  'p:now': 'Live values: a table of the current brain and vehicle numbers (phase, commitment, heading command, avoidance, altitude, body rates, object slots).',
  'p:model': 'Model: every vehicle parameter with its source, and a plain list of the simplifications this simulator makes.',
  'p:run': 'Recorded run: pick one of the study\'s pre-recorded runs, made by the Python reference. The replay never recomputes anything.',
  'p:mind': 'Brain readout: a plain-language sentence decoded from the brain\'s own state (top), with the simulator\'s truth beneath it, so you can see when the brain is wrong.',
  'p:view': 'View controls. Cameras: chase (behind the drone), top (overview, north up), free (drag to orbit). Quality: graphics detail (High, Balanced, Low) for slower computers. Display: the flight overlay with horizon, heading, speed and altitude (key H). Map: the small top-down map (key M). ' + VIEW,
};
const tipBox = document.createElement('div'); tipBox.className = 'htip'; tipBox.setAttribute('aria-hidden', 'true'); document.body.appendChild(tipBox);
let tipN = 0, tipPinned = null;
function helpBtn(text, label) {
  if (!text) return '';
  tipN++; return `<button type="button" class="hq" aria-label="Help: ${esc(label)}" aria-describedby="hlp${tipN}" aria-expanded="false" data-help="hlp${tipN}">?</button><span class="sr" id="hlp${tipN}">${esc(text)}</span>`;
}
function showTip(b) {
  const t = $(b.dataset.help); if (!t) return;
  const txt = t.textContent, i = txt.search(/(Part of the study|Exploratory|View only)/);
  tipBox.innerHTML = i > 0 ? `${esc(txt.slice(0, i))}<br><span class="tag">${esc(txt.slice(i))}</span>` : esc(txt);
  tipBox.classList.add('on');
  const r = b.getBoundingClientRect(), w = tipBox.offsetWidth, h = tipBox.offsetHeight;
  tipBox.style.left = clamp(r.left + r.width / 2 - w / 2, 8, innerWidth - w - 8) + 'px';
  tipBox.style.top = (r.bottom + h + 10 < innerHeight ? r.bottom + 6 : Math.max(8, r.top - h - 6)) + 'px';
}
function hideTip() { tipBox.classList.remove('on'); if (tipPinned) tipPinned.setAttribute('aria-expanded', 'false'); tipPinned = null; }
document.addEventListener('mouseover', (e) => { const b = e.target.closest && e.target.closest('.hq'); if (b && !tipPinned) showTip(b); });
document.addEventListener('mouseout', (e) => { const b = e.target.closest && e.target.closest('.hq'); if (b && !tipPinned) tipBox.classList.remove('on'); });
document.addEventListener('focusin', (e) => { if (e.target.classList && e.target.classList.contains('hq')) showTip(e.target); });
document.addEventListener('focusout', (e) => { if (e.target.classList && e.target.classList.contains('hq')) hideTip(); });
document.addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('.hq');
  if (!b) { if (tipPinned) hideTip(); return; }
  e.preventDefault(); e.stopPropagation(); // keep <summary> closed/open and labels from activating their control
  if (tipPinned === b) { hideTip(); return; }
  hideTip(); tipPinned = b; b.setAttribute('aria-expanded', 'true'); showTip(b);
}, true);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTip(); });
function addPanelHelp() { // dock headers, the recorded-run header, the brain readout and the view toolbar
  const map = [['Eye and memory', 'p:eye'], ['Steering hub', 'p:hub'], ['Home-vector memory', 'p:homev'], ['Attitude', 'p:att'], ['Live values', 'p:now'], ['Model', 'p:model'], ['Recorded run', 'p:run']];
  document.querySelectorAll('.dock summary, #runs summary').forEach((sm) => {
    const m = map.find(([k]) => sm.textContent.trim().startsWith(k)); if (!m) return;
    const who = sm.querySelector('.who'), html = helpBtn(HELP[m[1]], m[0]);
    if (who) who.insertAdjacentHTML('beforebegin', html); else sm.insertAdjacentHTML('beforeend', html);
  });
  const mk = document.querySelector('#mind .k'); if (mk) mk.insertAdjacentHTML('beforeend', helpBtn(HELP['p:mind'], 'Brain readout'));
  const tools = document.querySelector('#view .tools'); if (tools) tools.insertAdjacentHTML('afterbegin', `<div class="grp ovbox" style="align-self:center;padding:4px 6px">${helpBtn(HELP['p:view'], 'View controls')}</div>`);
}
// v2 fields: read app/sim/scenario.js's schema (re-checked every 30 s while open, as the brain code may update)
async function pollSupport() {
  if (!HTTP) return;
  try {
    const m = await import(new URL('sim/scenario.js?v=' + Date.now(), location.href).href);
    const bp = (((m.SCHEMA || {}).properties || {}).brain || {}).properties || {};
    const was = JSON.stringify(SUPPORT);
    SUPPORT.version = !!bp.version; SUPPORT.modules = !!bp.modules; SUPPORT.tip = !!bp.tip;
    if (JSON.stringify(SUPPORT) !== was) { syncKnobs(); writeHash(); }
  } catch (e) { console.warn('scenario.js schema not readable', e); }
}

// ---------------------------------------------------------------- theme, camera, resize, loop
function applyTheme() { readCss(); if (S3) S3.setTheme(document.documentElement.dataset.theme === 'light' ? 'light' : 'dark', css, roleColor()); panelDirty = true; if (mode === 'results') drawCharts(); }
$('theme').onclick = () => { const t = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'; document.documentElement.dataset.theme = t; try { localStorage.setItem('mbd-theme', t); } catch (e) { /* private */ } applyTheme(); };
const QDEF = matchMedia('(pointer: coarse)').matches || innerWidth < 900 ? 'balanced' : 'high';
function setQuality(q) {
  if (!['high', 'balanced', 'low'].includes(q)) q = QDEF;
  $('quality').value = q; $('sb-q').textContent = q; if (S3 && S3.setQuality) S3.setQuality(q);
}
$('quality').onchange = () => { setQuality($('quality').value); try { localStorage.setItem('mbd-quality', $('quality').value); } catch (e) { /* private */ } };
function setCam(m) { if (!S3) return; S3.setCam(m); document.querySelectorAll('#cams button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.cam === m))); }
document.querySelectorAll('#cams button').forEach((b) => { b.onclick = () => setCam(b.dataset.cam); });
$('mapbtn').onclick = () => { const w = $('mapwrap'); w.hidden = !w.hidden; $('mapbtn').setAttribute('aria-pressed', String(!w.hidden)); };
function resize() { if (S3) S3.resize(); panelDirty = true; if (mode === 'results' && results.list) drawCharts(); }
let roQ = false; new ResizeObserver(() => { if (roQ) return; roQ = true; requestAnimationFrame(() => { roQ = false; resize(); }); }).observe($('view'));
addEventListener('resize', () => { panelDirty = true; });

$('play').onclick = togglePlay; $('run').onclick = () => setRunning(!sim.running);
$('reset').onclick = $('reset2').onclick = resetSim; $('batch').onclick = runBatch;
$('speed').onchange = () => { if (sim.w) sim.w.postMessage({ cmd: 'speed', v: +$('speed').value }); };
$('copy').onclick = async () => {
  const txt = JSON.stringify(scenario(), null, 2), b = $('copy');
  try { await navigator.clipboard.writeText(txt); b.textContent = 'Copied'; } catch (e) { prompt('Scenario JSON', txt); }
  setTimeout(() => { b.textContent = 'Copy scenario'; }, 1400);
};
document.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toUpperCase();
  if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'range') || e.metaKey || e.ctrlKey || mode === 'paper') return;
  const k = e.key.toLowerCase();
  if (k === ' ') { togglePlay(); e.preventDefault(); }
  else if (k === 'r' && mode === 'sim') resetSim();
  else if (k === '1' || k === '2' || k === '3') setCam(['chase', 'top', 'free'][+k - 1]);
  else if (k === 'm') $('mapbtn').click();
  else if (k === 'h') $('hudbtn').click();
  else if (k === 't') $('theme').click();
});

let last = performance.now(), fpsN = 0, fpsT = performance.now(), fps = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (mode !== 'paper') {
    if (mode === 'results' && results.playing && results.run) { view.T = Math.min(view.dur, view.T + dt * +$('speed').value); if (view.T >= view.dur) { results.playing = false; playIcon(); } showRunFrame(); }
    if (S3) { S3.update(view.frame, view.trail, { present, arrows: true }); S3.render(); placeTags(); }
    drawHud(now);
    if (panelDirty && now - lastPanel > 33) { panelDirty = false; lastPanel = now; drawPanels(); drawTimeline(); status(); }
    fpsN++; if (now - fpsT > 500) { fps = (fpsN * 1000) / (now - fpsT); fpsN = 0; fpsT = now; status(); }
  }
  requestAnimationFrame(loop);
}
function status() {
  const f = view.frame;
  $('sb-t').textContent = (f ? f.t : 0).toFixed(2); $('sb-ph').textContent = f ? f.phase : '–';
  $('sb-arb').textContent = arbOf(); $('sb-fps').textContent = fps.toFixed(0);
  $('sb-rtf').textContent = mode === 'sim' ? `${sim.rtf.toFixed(2)}× (cap ${sim.cpu > 999 ? '>999' : sim.cpu.toFixed(0)}×)` : 'replay';
  $('sb-src').textContent = mode === 'sim' ? (sim.ok ? sim.src || '…' : 'needs http') : results.mockRun ? 'mock replay' : 'recorded';
}

// ---------------------------------------------------------------- boot
buildKnobs(); syncKnobs(); buildGallery(); modelPanel(); addPanelHelp();
let hud0 = null; try { hud0 = localStorage.getItem('mbd-hud'); } catch (e) { /* private */ }
setHud(hud0 !== '0', false);
pollSupport(); setInterval(() => { if (mode === 'sim' && !sim.running) pollSupport(); }, 30000);
if (!HTTP) noSim('Opened from file://');
const m0 = hashP().get('m');
setMode(m0 === 'results' || m0 === 'paper' ? m0 : 'sim');
setCam('chase');
let q0 = null; try { q0 = localStorage.getItem('mbd-quality'); } catch (e) { /* private */ }
setQuality(q0);
requestAnimationFrame(loop);
window.MBD = { get view() { return view; }, get K() { return K; }, scenario, setMode, sim };
})();
