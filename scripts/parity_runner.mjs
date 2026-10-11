// JS side of scripts/parity_check.py: reads {dt, T, every, scenarios} JSON on stdin, flies each scenario
// with app/sim/quad.js + dryden.js (batch 1), prints {name: {pos, euler, wind}} sampled every `every` steps.
import { Quad, Controller } from '../app/sim/quad.js';
import { Dryden } from '../app/sim/dryden.js';

const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const { dt, T, every, scenarios } = JSON.parse(Buffer.concat(chunks).toString());
const out = {};
for (const sc of scenarios) {
  const q = new Quad(undefined, sc.pos0, sc.yaw0), c = new Controller();
  const d = new Dryden(sc.seed, sc.turbulence, sc.mean_wind, sc.dryden_alt_m);
  const log = { pos: [], euler: [], wind: [] };
  const n = Math.round(T / dt);
  for (let k = 0; k < n; k++) {
    const t = k * dt;
    const sp = sc.schedule.filter(s => s.t <= t + 1e-9).at(-1);
    const wind = d.step(dt);
    q.step(c.update(q, sp, dt), wind, dt);
    if ((k + 1) % every === 0) { log.pos.push(Array.from(q.pos)); log.euler.push(q.euler()); log.wind.push(Array.from(wind)); }
  }
  out[sc.name] = log;
}
process.stdout.write(JSON.stringify(out));
