// Brain readout: template sentences (no language model), JS twin of scripts/brain/mind.py.
import { wrap } from './eye.js';

export const SECTORS = ['ahead', 'ahead-left', 'left', 'behind-left', 'behind', 'behind-right', 'right', 'ahead-right'];
const pmod = (x, m) => { const r = x % m; return (r !== 0 && (r < 0) !== (m < 0)) ? r + m : r; };
export const sector = (rel) => SECTORS[Math.floor(pmod(wrap(rel) + Math.PI / 8, 2 * Math.PI) / (Math.PI / 4)) % 8];

export function steeringSlot(headingCmd, bearings, strengths, visible) {
  let best = -1, bd = 30 * Math.PI / 180;
  for (let k = 0; k < bearings.length; k++) {
    const d = Math.abs(wrap(bearings[k] - headingCmd));
    if (visible[k] && strengths[k] > 0.2 && d < bd) { best = k; bd = d; }
  }
  return best;
}

const fmt0 = (x) => pyRound(x, 0), fmt2 = (x) => pyRound(x, 2);
// Python f"{x:.Nf}" rounds the exact binary value half-to-even; toFixed rounds the exact value half-up. They differ
// only on exact binary ties (e.g. 0.125 -> "0.12" vs "0.13"), so ties are resolved here as Python does.
function pyRound(x, n) {
  const s = x.toFixed(n), m = 10 ** n, y = x * m;
  if (Math.abs(y - Math.trunc(y)) === 0.5 && Number.isFinite(y)) {
    const lo = Math.floor(y), even = lo % 2 === 0 ? lo : lo + 1;
    return (even / m).toFixed(n);
  }
  return s;
}

export function say(phase, headingEst, homeEst, familiarity, slot, headingTrue, homeTrue, roles, objDist, faults) {
  const homeWords = (v, hd) => {
    const d = Math.hypot(v[0], v[1]);
    return d >= 1 ? `${fmt0(d)} m ${sector(Math.atan2(v[1], v[0]) - hd)}` : 'here';
  };
  let text = `home is ${homeWords(homeEst, headingEst)}; target familiarity ${fmt2(familiarity)}`;
  if (phase === 'home') text += '; delivered, flying home';
  else if (slot >= 0) text += `; committed to object ${slot}`;
  else text += '; searching';
  let truth = `truth: home ${homeWords(homeTrue, headingTrue)}`;
  if (slot >= 0 && phase !== 'home') truth += `; object ${slot} is ${roles[slot] === 'target' ? 'the target' : 'a look-alike'}, ${fmt0(objDist[slot])} m away`;
  if (faults.length) truth += '; faults: ' + faults.join(', ');
  return { text, truth };
}
