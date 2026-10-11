// MOCK summary for building the Results charts (workstream E). Synthetic numbers, NOT results.
// The integrator's scripts/export_app_summary.py writes data/summary.json + data/summary.js (window.APP_SUMMARY)
// with the same keys, plus a _source entry per number; the app prefers those and shows a MOCK DATA watermark otherwise.
// Shapes:
//   stubbornness_frontier: {arm: [{gain_or_knob, hold_s, hold_ci: [lo, hi], release_s, release_ci: [lo, hi]}]}
//   h1_success_vs_severity: {arm: [{severity, success, ci: [lo, hi]}]}      severity 0..1, success 0..1
//   recog_auroc: {method: {no_flight: {mean, lo, hi}, flight: {mean, lo, hi}}}  methods: mb, pixels, colour_hist, dinov2
window.MOCK_SUMMARY = (() => {
  const fr = (pts) => pts.map(([g, h, r]) => ({ gain_or_knob: g, hold_s: h, hold_ci: [h * 0.8, h * 1.2], release_s: r, release_ci: [r * 0.8, r * 1.25] }));
  const sev = (s0, k) => [0, 0.25, 0.5, 0.75, 1].map((s) => { const v = s0 * Math.exp(-k * s); return { severity: s, success: v, ci: [Math.max(0, v - 0.08), Math.min(1, v + 0.08)] }; });
  const au = (a, b) => ({ no_flight: { mean: a, lo: a - 0.05, hi: a + 0.05 }, flight: { mean: b, lo: b - 0.05, hi: b + 0.05 } });
  return {
    _mock: true,
    stubbornness_frontier: {
      ring: fr([[0, 0.4, 0.5], [0.5, 1.2, 0.9], [1, 2.4, 1.8], [1.5, 4.1, 3.6], [2, 6.0, 7.5]]),
      ring_static: fr([[0, 0.4, 0.5]]),
      vote: fr([[0, 0.2, 0.3]]),
      sum: fr([[0, 0.5, 0.7]]),
      accumulator: fr([[0.5, 1.0, 0.8], [1, 2.2, 1.6], [2, 3.9, 3.2], [4, 5.6, 6.2]]),
      state_machine: fr([[0, 3.0, 2.6]]),
    },
    h1_success_vs_severity: { ring: sev(0.92, 0.9), ring_static: sev(0.9, 1.2), vote: sev(0.85, 1.4), sum: sev(0.88, 1.1), accumulator: sev(0.91, 0.95), state_machine: sev(0.9, 0.85) },
    recog_auroc: { mb: au(0.66, 0.78), pixels: au(0.64, 0.72), colour_hist: au(0.58, 0.6), dinov2: au(0.8, 0.84) },
  };
})();
