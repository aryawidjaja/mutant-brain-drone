// Live simulator worker (module worker; needs http, e.g. `npx serve app`). Steps the C6 mission at a fixed brain tick
// (0.02 s, 10 physics steps inside the mission) at real time x speed, independent of the frame rate, and posts the
// latest frame at most 20 times a second (frame() ray-casts the eye panorama, ~6 ms in a dense forest). Uses
// app/sim/mission.js (+ app/sim/world.js) when present, else the clearly labelled mock in app/data/.
//
// in:  {cmd:'init', scenario, speed, opts}  {cmd:'run'}  {cmd:'pause'}  {cmd:'speed', v}  {cmd:'step'}  {cmd:'batch', scenario, seeds:[...], opts}
//      opts = {yawSurrogate: {fit, k}} (exploratory, passed to createMission; {} = the reference behaviour)
// out: {type:'ready', world, source:'real'|'mock', note}  {type:'frame', frame, rtf, cpu}  {type:'done', outcome}
//      {type:'progress', i, n, seed, outcome}  {type:'batch', results}  {type:'error', message}
const TICK = 0.02, POST_MS = 50;
let M = null, running = false, speed = 1, last = 0, simAcc = 0, rtf = 0, cpu = 0, timer = 0, sentDone = false, lastPost = 0, fresh = false;

async function loadWorld() {
  try { if ((await fetch('./sim/world.js', { method: 'HEAD', cache: 'no-store' })).ok) return await import('./sim/world.js'); } catch (e) { console.warn('world.js', e); }
  return null;
}
async function load() {
  let note = '';
  try {
    const head = await fetch('./sim/mission.js', { method: 'HEAD', cache: 'no-store' });
    if (head.ok) {
      const mod = await import('./sim/mission.js');
      if (typeof mod.createMission !== 'function') throw new Error('app/sim/mission.js has no createMission export');
      return { mod, wmod: await loadWorld(), source: 'real', note };
    }
    note = 'app/sim/mission.js not found';
  } catch (e) { note = 'app/sim/mission.js failed to load: ' + e.message; }
  return { mod: await import('./data/mock_mission.js'), wmod: await loadWorld(), source: 'mock', note };
}
let L = null;
const bw0 = () => (L && L.wmod ? 'app/sim/world.js' : 'mock');
async function build(sc, opts) {
  L = L || await load();
  const bw = L.wmod && (L.wmod.buildWorld || L.wmod.build_world);
  const m = L.source === 'mock' ? L.mod.createMission(sc, bw ? bw(sc) : null) : L.mod.createMission(sc, opts || {});
  const world = m.world || (bw ? bw(sc) : null);
  return { m, world };
}

function tick() {
  if (!M) return;
  const now = performance.now(), wall = Math.min(0.1, (now - last) / 1000); last = now;
  let stepped = 0;
  if (running) {
    simAcc += wall * speed;
    const t0 = performance.now();
    while (simAcc >= TICK && !M.done && performance.now() - t0 < 12) { M.step(); simAcc -= TICK; stepped++; }
    if (simAcc > 0.25) simAcc = 0.25; // fell behind: drop time rather than spiral
    const used = (performance.now() - t0) / 1000;
    rtf = 0.95 * rtf + 0.05 * (stepped * TICK / Math.max(wall, 1e-3)); // every tick, so idle ticks count too
    if (stepped) cpu = 0.9 * cpu + 0.1 * (stepped * TICK / Math.max(used, 1e-6));
  }
  if (stepped) fresh = true;
  if (fresh && (now - lastPost >= POST_MS || M.done)) post();
}
function post() {
  fresh = false; lastPost = performance.now();
  const f = M.frame();
  if (!f) return; // nothing stepped yet
  if (!f.outcome && M.done && M.finish) f.outcome = M.finish(); // time limit: the last logged tick predates the end
  postMessage({ type: 'frame', frame: f, rtf, cpu, running });
  if (f.outcome && !sentDone) { sentDone = true; running = false; postMessage({ type: 'done', outcome: f.outcome }); }
}

onmessage = async (e) => {
  const d = e.data;
  try {
    if (d.cmd === 'init') {
      const { m, world } = await build(d.scenario, d.opts);
      M = m; speed = d.speed || 1; running = false; simAcc = 0; rtf = 0; sentDone = false; last = performance.now();
      postMessage({ type: 'ready', world, source: L.source, note: L.note, worldSrc: bw0() });
      post();
      clearInterval(timer); timer = setInterval(tick, 1000 / 60);
    } else if (d.cmd === 'run') { running = !sentDone; last = performance.now(); }
    else if (d.cmd === 'pause') running = false;
    else if (d.cmd === 'speed') speed = d.v;
    else if (d.cmd === 'step' && M) { for (let i = 0; i < 5; i++) M.step(); post(); }
    else if (d.cmd === 'batch') {
      // exploratory only: run each seed to its end as fast as possible
      const results = [];
      for (let i = 0; i < d.seeds.length; i++) {
        const sc = { ...d.scenario, seed: d.seeds[i] }, { m } = await build(sc, d.opts), nmax = Math.ceil(sc.mission.max_time_s / TICK) + 1;
        let out;
        if (typeof m.finish === 'function') { for (let k = 0; k < nmax && !m.done; k++) m.step(); out = m.finish(); }
        else { // mock: only frames carry the outcome
          let f = null;
          for (let k = 0; k < nmax; k++) { m.step(); if (k % 25 === 0 || k === nmax - 1) { f = m.frame(); if (f.outcome) break; } }
          out = f.outcome || { timeout: true };
        }
        results.push({ seed: sc.seed, ...out });
        postMessage({ type: 'progress', i: i + 1, n: d.seeds.length, seed: sc.seed, outcome: out });
      }
      postMessage({ type: 'batch', results, source: L.source });
    }
  } catch (err) { postMessage({ type: 'error', message: String(err && err.stack || err) }); }
};
