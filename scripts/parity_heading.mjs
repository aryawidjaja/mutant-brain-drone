// Node side of scripts/brain/heading.py:js_parity. Usage: node scripts/parity_heading.mjs in.json out.json
import { readFileSync, writeFileSync } from 'node:fs';
import { CLASSES } from '../app/sim/brain/heading.js';

const s = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const est = new CLASSES[s.name](s.params);
const heading = [], conf = [];
for (let k = 0; k < s.gyro.length; k++) {
  heading.push(est.step(s.gyro[k], s.sun[k], s.valid[k], s.dt));
  conf.push(est.conf);
}
writeFileSync(process.argv[3], JSON.stringify({ heading, conf }));
