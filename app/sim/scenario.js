// Scenario contract C1 (JS twin of scripts/scenario.py): validate, fill defaults, hash, sample.
// The schema below is a verbatim copy of schema/scenario.schema.json (scenario.py's self-check asserts it).
import { Rng } from './rng.js';

export const SCHEMA = /*SCHEMA*/{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"mutant-brain-drone/scenario.schema.json","title":"Mutant Brain Drone scenario (contract 1; world and wind compatible with silent-flock)","type":"object","required":["seed","world","home","target","wind","brain","mission"],"additionalProperties":false,"properties":{"seed":{"type":"integer","minimum":0},"world":{"type":"object","required":["size_m","landscape","obstacle_density"],"additionalProperties":false,"properties":{"size_m":{"type":"array","items":{"type":"number","exclusiveMinimum":0},"minItems":2,"maxItems":2},"landscape":{"enum":["meadow","forest","urban"]},"obstacle_density":{"type":"number","minimum":0,"maximum":1},"light":{"type":"number","minimum":0,"maximum":1,"default":1.0},"fog_m":{"type":["number","null"],"exclusiveMinimum":0,"default":null}}},"home":{"$ref":"#/$defs/vec3"},"target":{"type":"object","required":["kind","pos"],"additionalProperties":false,"properties":{"kind":{"enum":["tent","pad","backpack"]},"pos":{"$ref":"#/$defs/vec3"},"yaw_deg":{"type":"number","default":0}}},"decoys":{"type":"array","default":[],"items":{"type":"object","required":["kind","pos"],"additionalProperties":false,"properties":{"kind":{"enum":["tent","pad","backpack"]},"pos":{"$ref":"#/$defs/vec3"},"swap":{"type":"boolean","default":false},"similarity":{"type":"number","minimum":0,"maximum":1,"default":0.8},"t_s":{"type":"number","minimum":0,"default":0}}}},"wind":{"type":"object","required":["mean_mps","turbulence"],"additionalProperties":false,"properties":{"mean_mps":{"$ref":"#/$defs/vec3"},"turbulence":{"enum":["none","light","moderate","severe"]},"shared_fraction":{"type":"number","minimum":0,"maximum":1,"default":1.0}}},"faults":{"type":"array","default":[],"items":{"type":"object","required":["kind","t_s"],"additionalProperties":false,"properties":{"kind":{"enum":["occlusion","compass_half","eye_columns","false_sun","speed_scale"]},"t_s":{"type":"number","minimum":0},"params":{"type":"object","default":{}}}}},"brain":{"type":"object","required":["arbiter"],"additionalProperties":false,"properties":{"arbiter":{"enum":["ring","ring_static","vote","sum","accumulator","state_machine"]},"ring_gain":{"type":"number","minimum":0,"default":1.0},"learning_flight_s":{"type":"number","minimum":0,"default":10},"params":{"type":"object","default":{},"description":"frozen tuned parameters for the chosen arbiter (results/tuned_params.json)"}}},"mission":{"type":"object","required":["max_time_s","dt_s"],"additionalProperties":false,"properties":{"max_time_s":{"type":"number","exclusiveMinimum":0},"dt_s":{"type":"number","exclusiveMinimum":0}}}},"$defs":{"vec3":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}}}/*END*/;
export const FAULTS = ['occlusion', 'decoy', 'compass_half', 'eye_columns', 'false_sun', 'speed_scale', 'gust'];
export const ARBITERS = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine'];

const clone = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));

function check(v, s, path, root) {
  if (s.$ref) s = root.$defs[s.$ref.split('/').pop()];
  if (s.type !== undefined) {
    const types = Array.isArray(s.type) ? s.type : [s.type];
    const isObj = v !== null && typeof v === 'object' && !Array.isArray(v);
    const ok = { object: isObj, array: Array.isArray(v), string: typeof v === 'string', boolean: typeof v === 'boolean',
      null: v === null, integer: Number.isInteger(v), number: typeof v === 'number' && Number.isFinite(v) };
    if (!types.some((t) => ok[t])) throw new Error(`${path}: expected ${s.type}, got ${JSON.stringify(v)}`);
  }
  if (s.enum && !s.enum.includes(v)) throw new Error(`${path}: ${JSON.stringify(v)} not in ${s.enum}`);
  if (typeof v === 'number') {
    if (s.minimum !== undefined && v < s.minimum) throw new Error(`${path}: ${v} < ${s.minimum}`);
    if (s.maximum !== undefined && v > s.maximum) throw new Error(`${path}: ${v} > ${s.maximum}`);
    if (s.exclusiveMinimum !== undefined && v <= s.exclusiveMinimum) throw new Error(`${path}: ${v} <= ${s.exclusiveMinimum}`);
  }
  if (Array.isArray(v)) {
    if (v.length < (s.minItems ?? 0) || v.length > (s.maxItems ?? 1 << 30)) throw new Error(`${path}: length ${v.length} out of range`);
    return s.items ? v.map((x, i) => check(x, s.items, `${path}[${i}]`, root)) : v.slice();
  }
  if (v !== null && typeof v === 'object') {
    const props = s.properties || {};
    for (const k of s.required || []) if (!(k in v)) throw new Error(`${path}: missing required '${k}'`);
    if (s.additionalProperties === false) {
      const extra = Object.keys(v).filter((k) => !(k in props));
      if (extra.length) throw new Error(`${path}: unknown keys ${extra}`);
    }
    const out = {};
    for (const [k, ps] of Object.entries(props)) {
      if (k in v) out[k] = check(v[k], ps, `${path}.${k}`, root);
      else if ('default' in ps) out[k] = clone(ps.default);
    }
    for (const k of Object.keys(v)) if (!(k in out)) out[k] = clone(v[k]);
    return out;
  }
  return v;
}

/** object or JSON string -> validated scenario with defaults filled (throws Error). */
export function load(s) {
  if (typeof s === 'string') s = JSON.parse(s);
  return check(s, SCHEMA, 'scenario', SCHEMA);
}

function canon(v) {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'number') return String(Math.floor(v * 1e6 + 0.5));
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
  return '{' + Object.keys(v).sort().map((k) => JSON.stringify(k) + ':' + canon(v[k])).join(',') + '}';
}

async function sha256hex(text) {
  const buf = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Condition hash (seed excluded), 12 hex chars; async because WebCrypto is. */
export async function scenarioHash(s) {
  s = load(strip(s));
  const body = {};
  for (const k of Object.keys(s)) if (k !== 'seed') body[k] = s[k];
  return (await sha256hex(canon(body))).slice(0, 12);
}

export function strip(s) {
  const o = {};
  for (const k of Object.keys(s)) if (!k.startsWith('_')) o[k] = s[k];
  return o;
}

export function faultParams(fault, sev) {
  return {
    occlusion: { duration_s: 1 + 9 * sev },
    decoy: { similarity: 0.5 + 0.45 * sev },
    compass_half: {},
    eye_columns: { frac: 0.1 + 0.3 * sev },
    false_sun: { offset_deg: 30 + 90 * sev },
    speed_scale: { dev: 0.3 * sev },
    gust: { level: ['light', 'moderate', 'severe'][Math.min(2, Math.floor(sev * 3))] },
  }[fault];
}

/** Mirror of scenario.py sample_scenario (same draw order on Rng(seed, 5)). */
export function sampleScenario(seed, { split = 'train', fault = null, severity = null, arbiter = 'ring', ring_gain = 1.0,
  learning_flight_s = 10.0, landscape = 'meadow', obstacle_density = 0.3, params = null, max_time_s = 300.0,
  dist_m = [25.0, 40.0], checkSplit = true } = {}) {
  if (checkSplit && !(split === 'train' ? seed >= 0 && seed <= 99 : seed >= 1000 && seed <= 1299))
    throw new Error(`seed ${seed} not in ${split} split`);
  const r = new Rng(seed, 5);
  const u = Array.from({ length: 10 }, () => r.uniform());
  const d = dist_m[0] + (dist_m[1] - dist_m[0]) * u[0];
  const b = 2 * Math.PI * u[1];
  const tpos = [d * Math.cos(b), d * Math.sin(b), 0.0];
  fault = fault === null ? FAULTS[Math.min(6, Math.floor(u[3] * 7))] : fault;
  const sev = severity === null ? u[4] : Number(severity);
  const s = {
    seed, world: { size_m: [100.0, 100.0], landscape, obstacle_density, light: 1.0, fog_m: null },
    home: [0.0, 0.0, 0.0], target: { kind: 'tent', pos: tpos, yaw_deg: 360 * u[2] },
    decoys: [], wind: { mean_mps: [0.0, 0.0, 0.0], turbulence: 'none', shared_fraction: 1.0 }, faults: [],
    brain: { arbiter, ring_gain, learning_flight_s, params: params || {} },
    mission: { max_time_s, dt_s: 0.002 },
  };
  const tOn = 5 + 20 * u[5];
  if (fault !== 'none') {
    const p = faultParams(fault, sev);
    if (fault === 'decoy') {
      const a = 2 * Math.PI * u[6], off = 8 + 6 * u[7];
      s.decoys = [{ kind: 'tent', pos: tpos.slice(), swap: true, similarity: p.similarity, t_s: 0.0 }];
      s.target.pos = [tpos[0] + off * Math.cos(a), tpos[1] + off * Math.sin(a), 0.0];
    } else if (fault === 'gust') {
      const a = 2 * Math.PI * u[8];
      s.wind = { mean_mps: [1.5 * Math.cos(a), 1.5 * Math.sin(a), 0.0], turbulence: p.level, shared_fraction: 1.0 };
    } else if (fault === 'occlusion') {
      s.faults = [{ kind: 'occlusion', t_s: 0.0, params: p }];
    } else if (fault === 'speed_scale') {
      s.faults = [{ kind: 'speed_scale', t_s: tOn, params: { scale: 1 + (u[6] < 0.5 ? p.dev : -p.dev) } }];
    } else if (fault === 'eye_columns') {
      s.faults = [{ kind: 'eye_columns', t_s: tOn, params: { ...p, start_col: Math.floor(u[9] * 72) } }];
    } else {
      s.faults = [{ kind: fault, t_s: tOn, params: p }];
    }
  }
  const out = load(s);
  out._meta = { split, fault, severity: sev };
  return out;
}
