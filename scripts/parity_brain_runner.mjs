// Node side of scripts/parity_brain.py: reads a spec JSON (inputs recorded by Python), runs the JS brain modules and
// mission on them, writes the JS outputs. Usage: node scripts/parity_brain_runner.mjs spec.json out.json
import { readFileSync, writeFileSync } from 'node:fs';
import { sampleScenario, scenarioHash, load } from '../app/sim/scenario.js';
import { buildWorld } from '../app/sim/world.js';
import * as eye from '../app/sim/brain/eye.js';
import { MushroomBody, ranks } from '../app/sim/brain/mb.js';
import { PathIntegrator } from '../app/sim/brain/pi.js';
import { reflex } from '../app/sim/brain/avoid.js';
import { say } from '../app/sim/brain/mind.js';
import { make } from '../app/sim/brain/arbiters.js';
import { createMission, DT_B } from '../app/sim/mission.js';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = {};

// 1. scenarios + hashes
out.scenarios = [];
for (const c of spec.scenarios) {
  const s = sampleScenario(c.seed, { ...c.kw, checkSplit: true });
  out.scenarios.push({ scenario: s, hash: await scenarioHash(s) });
}
// 2. worlds
out.worlds = spec.worlds.map((sc) => buildWorld(load(sc)));
// 3. panoramas
out.panoramas = spec.panoramas.map((p) => {
  const S = new eye.Scene(buildWorld(load(p.scenario)));
  return Array.from(eye.panorama(S, p.pos, p.yaw, p.t, p.dead, p.false_sun));
});
// 4. looks + 5. MB familiarity on Python's recorded looks and on JS's own looks
out.looks = [];
out.mb = [];
const mbBase = new MushroomBody(spec.mb.n_in, 20000, 10, 0.01, 0);
for (const L of spec.looks) {
  const S = new eye.Scene(buildWorld(load(L.scenario)));
  const own = L.poses.map((p) => Array.from(eye.look(S, L.slot, p, L.t)[0]));
  const learnOwn = eye.learningLooks(S, L.learning_flight_s).map((g) => Array.from(g));
  out.looks.push({ looks: own, learn: learnOwn });
  const famOn = (learn, probes) => {
    const m = Object.create(mbBase); m.reset();
    for (const g of learn) m.learn(ranks(g));
    return probes.map((g) => m.familiarity(ranks(g)));
  };
  out.mb.push({ recorded: famOn(L.py_learn, L.py_probes), own: famOn(learnOwn, own.map((g, i) => g.map((v, j) => v + eye.NOISE * L.noise[i][j]))) });
}
out.mb_first_idx = Array.from(mbBase.idx.slice(0, 25));
// 6. path integrator on a recorded input sequence
{
  const p = new PathIntegrator();
  const rec = [];
  spec.pi.theta.forEach((th, i) => {
    for (let k = 0; k < 16; k++) p.colMask[k] = spec.pi.mask[i][k];
    const h = p.step(th, spec.pi.v[i], spec.pi.dt);
    if (i % 50 === 49) rec.push([h, ...p.homeVector()]);
  });
  out.pi = rec;
}
// 7. arbiters on a recorded obs sequence
out.arbiters = {};
for (const [name, params] of Object.entries(spec.arbiters.params)) {
  const a = make(name, params);
  out.arbiters[name] = spec.arbiters.obs.map((o) => { const r = a.step(o, DT_B); return [r.heading_cmd, r.commit]; });
}
// 8. avoidance
out.avoid = spec.avoid.map((c) => { const r = reflex(c.depth, c.v, c.noise, c.dead); return [r.yawOffset, r.speedFactor, r.loom]; });
// 9. mind
out.mind = spec.mind.map((a) => { const r = say(...a); return [r.text, r.truth]; });
// 10. missions: frames every 5 ticks (10 Hz) up to T, plus the outcome
out.missions = spec.missions.map((c) => {
  const m = createMission(c.scenario);
  const fr = [];
  while (m.t < c.T - 1e-9 && !m.done) {
    const tick = Math.round(m.t / DT_B);
    m.step();
    if (tick % 5 === 0) {
      const f = m.frame();
      fr.push({ t: f.t, pos: f.quad.pos, phase: f.phase, heading_cmd_deg: f.ring.heading_cmd_deg, familiarity: f.familiarity,
        home_est: f.pi.home_est, eye: c.eye ? Array.from(f.eye.rgb) : null, mind: f.mind.text });
    }
  }
  return { frames: fr, outcome: m.finish() };
});
writeFileSync(process.argv[3], JSON.stringify(out));
