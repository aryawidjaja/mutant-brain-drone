// Seeded PRNG, bit-identical to scripts/rng.py (mulberry32 + Box-Muller). Batch size 1.
export function streamSeed(seed, stream) {
  return (Number((BigInt(seed) * 0x9E3779B1n + BigInt(stream) * 0x85EBCA77n + 0x6D2B79F5n) & 0xFFFFFFFFn)) >>> 0;
}

export class Rng {
  constructor(seed, stream = 0) { this.a = streamSeed(seed, stream); }
  uniform() {
    this.a = (this.a + 0x6D2B79F5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal() {
    const u1 = 1 - this.uniform(), u2 = this.uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// node app/sim/rng.js prints the same reference values as `uv run python scripts/rng.py`
if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  const r = new Rng(42, 0);
  const ref = [r.uniform(), r.uniform(), r.uniform(), new Rng(42, 5).normal()];
  console.log(ref.map(v => v.toFixed(12)).join(' '));
}
