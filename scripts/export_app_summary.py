"""Export every number the app's Results tab and the web paper (app/guide.js) show into one auditable file.

  uv run python scripts/export_app_summary.py     # -> app/data/summary.json + app/data/summary.js (window.APP_SUMMARY)

Reads whatever result files exist in results/. Optional files that are not there yet (the 3D mission: mission_h1.json,
mission_h3.json, runs/mission_test.parquet; parity_brain.json) leave their block null, and the paper prints "pending";
re-running this script after they land fills them. Every exported leaf has an entry in the parallel "_source" map:
"<result file>:<key/path>" for a copied value (path separator "/", list indices as integers) or "derived: <formula>" for
a value computed here. The self-check (always run at the end) re-reads every file source and asserts equality, re-derives
the derived values from the raw run records, and asserts that no leaf lacks a source.

Expected shapes of the optional 3D files (so the integrator can match them):
  results/mission_h1.json  same keys as results/kt2.json: n_seeds, aggregate_auc{arm:{mean,lo,hi}}, ring_minus_best_other
                           {est,lo,hi,p}, best_other_point, ring_minus_each{arm:{est,lo,hi,p,p_holm}}, per_fault_auc{arm:
                           {fault:auc}}, success_by_condition{arm:{"none:None", "<fault>:<severity>", ...}} (levels in
                           ascending order), worst{arm:{fault,auc,worst_condition,worst_condition_success}},
                           wrong_object_stop_rate{arm}, t_hold_occlusion_mean_s{arm}, t_release_decoy_mean_s{arm},
                           tuned_params{arm:{...}}, gate, provenance.
  results/runs/mission_test.parquet   C7 columns + fault, level (as kt2_test.parquet); used for the aggregate-curve CI.
  results/mission_h3.json  same keys as results/kt3.json: t_cross_s, points{arm:[{knob,value,hold_mean_s,release_mean_s,
                           hold_sem,release_sem}]}, ring_vs_rival{state_machine|accumulator:[{ring_gain,delta_s,lo,hi}]},
                           posthoc_dominance_frontier{delta{...same...}}, spearman_vs_gain{hold,release}, gate,
                           dead_because, provenance.
  results/parity_brain.json   {pass, results:{missions:[{max_pos_err_first_20s_m, ...}], ...}} (as written by WS-D).
"""
import json, math, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'app' / 'data'
REPO_URL = 'https://github.com/aryawidjaja/mutant-brain-drone'   # planned; not public until the owner approves
ARMS = ['ring', 'ring_static', 'vote', 'sum', 'accumulator', 'state_machine']
Z = 1.959963984540054   # two-sided 95% normal quantile
RECOG = {'mb': 'mb', 'pixels': 'pixels', 'colour': 'colour', 'dino_patch': 'dino_patch', 'dino_cls': 'dino_cls',
         'mb_edge': 'mb_edge', 'pixels_crop': 'pixels_crop', 'mb_crop': 'mb_crop'}
APP_RECOG = {'mb': 'mb', 'pixels': 'pixels', 'colour_hist': 'colour', 'dinov2': 'dino_patch'}   # app key -> file key


def _clean(x):
    """JSON-safe: numpy -> python, nan -> null, inf -> 'inf'."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) else ('inf' if x == math.inf else '-inf' if x == -math.inf else x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Out:
    def __init__(self):
        self.d, self.src = {}, {}

    def put(self, path, value, src):
        node, keys = self.d, path.split('.')
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _clean(value)
        self.src[path] = src

    def copy(self, path, fname, keypath, data):
        """Copy data[keypath] (keypath '/'-separated) from results/<fname> to summary path."""
        self.put(path, get(data, keypath), f'results/{fname}:{keypath}')


def listify(x):
    """Dicts built by Out.put with keys '0'..'n-1' become lists (paths like 'a.0.b' still resolve through get())."""
    if isinstance(x, dict):
        x = {k: listify(v) for k, v in x.items()}
        if x and all(k.isdigit() for k in x) and sorted(map(int, x)) == list(range(len(x))):
            return [x[str(i)] for i in range(len(x))]
    return x


def get(d, keypath, sep='/'):
    for k in keypath.split(sep):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def wilson(p, n):
    """Wilson 95% interval for a proportion p = k/n."""
    k = round(p * n)
    ph, den = k / n, 1 + Z ** 2 / n
    c, h = (ph + Z ** 2 / (2 * n)) / den, Z * math.sqrt(ph * (1 - ph) / n + Z ** 2 / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def conditions(sbc):
    """success_by_condition keys -> {fault: [(severity_label, key), ...]} in file order (levels ascending)."""
    out = {}
    for key in sbc:
        f, s = key.split(':', 1)
        if f != 'none':
            out.setdefault(f, []).append((s, key))
    return out


GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def aggregate_curve(runs, arm):
    """Per seed: success of each fault interpolated onto GRID (x = level / n_levels, level 0 = no fault), mean over
    faults; then mean over seeds and a 95% percentile bootstrap over seeds (2,000 resamples, numpy seed 0)."""
    r = runs[runs.arbiter == arm].copy()
    r['succ'] = (r.success_target & r.success_home).astype(float)
    base = r[r.fault == 'none'].set_index('seed').succ
    per = []
    for f, g in r[r.fault != 'none'].groupby('fault'):
        L = g.level.max()
        tab = g.pivot(index='seed', columns='level', values='succ').reindex(columns=range(1, L + 1))
        tab.insert(0, 0, base.reindex(tab.index))
        xs = np.arange(L + 1) / L
        per.append(np.stack([np.interp(GRID, xs, row) for row in tab.to_numpy()]))
    v = np.mean(per, axis=0)                      # [seeds, grid]
    idx = np.random.default_rng(0).integers(0, len(v), (2000, len(v)))
    bm = v[idx].mean(1)                            # [boot, grid]
    return v.mean(0), np.percentile(bm, 2.5, 0), np.percentile(bm, 97.5, 0)


# ---------------------------------------------------------------- blocks
def h1_block(o, pre, fname, d, runs_name):
    """KT2 (2D pilot) or the 3D mission factorial: same keys."""
    for k in ['n_seeds', 'best_other_point', 'gate']:
        o.copy(f'{pre}.{k}', fname, k, d)
    for k in ['est', 'lo', 'hi', 'p']:
        o.copy(f'{pre}.ring_minus_best_other.{k}', fname, f'ring_minus_best_other/{k}', d)
    n = d['n_seeds']
    sbc0 = d['success_by_condition'][ARMS[0]]
    cond = conditions(sbc0)
    o.put(f'{pre}.faults', list(cond), f'results/{fname}:success_by_condition/{ARMS[0]} (keys, file order)')
    for a in ARMS:
        for k in ['mean', 'lo', 'hi']:
            o.copy(f'{pre}.aggregate_auc.{a}.{k}', fname, f'aggregate_auc/{a}/{k}', d)
        if a != 'ring':
            for k in ['est', 'lo', 'hi', 'p', 'p_holm']:
                o.copy(f'{pre}.ring_minus_each.{a}.{k}', fname, f'ring_minus_each/{a}/{k}', d)
        for k in ['fault', 'auc', 'worst_condition', 'worst_condition_success']:
            o.copy(f'{pre}.worst.{a}.{k}', fname, f'worst/{a}/{k}', d)
        for k in ['wrong_object_stop_rate', 't_hold_occlusion_mean_s', 't_release_decoy_mean_s']:
            o.copy(f'{pre}.{k}.{a}', fname, f'{k}/{a}', d)
        o.copy(f'{pre}.success_none.{a}', fname, f'success_by_condition/{a}/none:None', d)
        for f, levels in cond.items():
            o.copy(f'{pre}.per_fault_auc.{a}.{f}', fname, f'per_fault_auc/{a}/{f}', d)
            o.put(f'{pre}.levels.{f}', [s for s, _ in levels], f'results/{fname}:success_by_condition/{ARMS[0]} (severity part of the keys)')
            for i, (s, key) in enumerate(levels):
                o.copy(f'{pre}.curves.{a}.{f}.{i}.success', fname, f'success_by_condition/{a}/{key}', d)
                o.put(f'{pre}.curves.{a}.{f}.{i}.ci', wilson(d['success_by_condition'][a][key], n),
                      f'derived: Wilson 95% interval, k = round(success * n_seeds), n_seeds = {n}')
            o.put(f'{pre}.none_ci.{a}', wilson(d['success_by_condition'][a]['none:None'], n),
                  f'derived: Wilson 95% interval, k = round(success * n_seeds), n_seeds = {n}')
    for a, ks in {'ring': ['ring_gain', 'tau_s', 'g_home'], 'ring_static': ['tau_s'], 'accumulator': ['tau_leak_s'],
                  'state_machine': ['timeout_s']}.items():
        for k in ks:
            if k in d['tuned_params'][a]:
                o.copy(f'{pre}.tuned.{a}.{k}', fname, f'tuned_params/{a}/{k}', d)
    o.put(f'{pre}.n_params', {a: len(d['tuned_params'][a]) for a in ARMS}, f'derived: len(results/{fname}:tuned_params/<arm>)')
    o.copy(f'{pre}.git', fname, 'provenance/git', d)
    rp = RES / 'runs' / runs_name
    if rp.exists():
        runs = pd.read_parquet(rp)
        o.put(f'{pre}.n_runs', len(runs), f'derived: rows of results/runs/{runs_name}')
        for a in ARMS:
            m, lo, hi = aggregate_curve(runs, a)
            o.put(f'{pre}.aggregate_curve.{a}', [dict(severity=x, success=y, ci=[l, h]) for x, y, l, h in zip(GRID, m, lo, hi)],
                  f'derived: results/runs/{runs_name}, per seed mean over faults of success interpolated at severity {GRID} '
                  '(x = level/n_levels), mean over seeds, 95% percentile bootstrap over seeds (2,000, seed 0)')


def h3_block(o, pre, fname, d):
    o.copy(f'{pre}.t_cross_s', fname, 't_cross_s', d)
    o.copy(f'{pre}.gate', fname, 'gate', d)
    o.copy(f'{pre}.dead_because', fname, 'dead_because', d)
    o.copy(f'{pre}.posthoc_gate', fname, 'posthoc_dominance_frontier/gate', d)
    o.copy(f'{pre}.posthoc_dead_because', fname, 'posthoc_dominance_frontier/dead_because', d)
    for k in ['hold', 'release']:
        o.copy(f'{pre}.spearman.{k}', fname, f'spearman_vs_gain/{k}', d)
    for a in ARMS:
        for i, p in enumerate(d['points'][a]):
            for k in ['knob', 'value', 'hold_mean_s', 'release_mean_s', 'hold_sem', 'release_sem']:
                o.copy(f'{pre}.points.{a}.{i}.{k}', fname, f'points/{a}/{i}/{k}', d)
            for k in ['hold', 'release']:
                m, s = p[f'{k}_mean_s'], p[f'{k}_sem']
                o.put(f'{pre}.points.{a}.{i}.{k}_ci', [m - Z * s, m + Z * s], f'derived: {k}_mean_s ± 1.96 × {k}_sem (normal approximation over seeds)')
    for r in ['state_machine', 'accumulator']:
        for i in range(len(d['ring_vs_rival'][r])):
            for k in ['ring_gain', 'delta_s', 'lo', 'hi']:
                o.copy(f'{pre}.delta.{r}.{i}.{k}', fname, f'ring_vs_rival/{r}/{i}/{k}', d)
                o.copy(f'{pre}.delta_posthoc.{r}.{i}.{k}', fname, f'posthoc_dominance_frontier/delta/{r}/{i}/{k}', d)
    o.copy(f'{pre}.git', fname, 'provenance/git', d)


def app_keys(o, h1pre, h3pre, h1from, h3from):
    """The Results tab's three charts (shapes fixed by app/app.js and app/panels.js)."""
    H3 = get(o.d, h3pre, '.')
    for a in ARMS:
        o.put(f'stubbornness_frontier.{a}', [dict(gain_or_knob=p['value'], hold_s=p['hold_mean_s'], hold_ci=p['hold_ci'],
              release_s=p['release_mean_s'], release_ci=p['release_ci']) for p in H3['points'][a]],
              f'derived: copy of {h3pre}.points.{a} (value, hold_mean_s, hold_ci, release_mean_s, release_ci)')
    o.put('stubbornness_frontier_from', h3from, 'derived: which file fed stubbornness_frontier')
    H1 = get(o.d, h1pre, '.')
    if 'aggregate_curve' in H1:
        for a in ARMS:
            o.put(f'h1_success_vs_severity.{a}', H1['aggregate_curve'][a], f'derived: copy of {h1pre}.aggregate_curve.{a}')
        o.put('h1_success_vs_severity_from', h1from, 'derived: which file fed h1_success_vs_severity')
    R = o.d['h4']['rec']
    for app, fk in APP_RECOG.items():
        for lf, k in [('no_flight', 'lf_off'), ('flight', 'lf_on')]:
            v = R[fk][k]
            o.put(f'recog_auroc.{app}.{lf}', dict(mean=v['auroc'][0], lo=v['lo'][0], hi=v['hi'][0]),
                  f'derived: AUROC at 0° = h4.rec.{fk}.{k}.(auroc|lo|hi)[0]')


def build():
    o = Out()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    o.put('meta.generated', now, 'derived: export time (UTC)')
    o.put('meta.repo_url', REPO_URL, 'derived: planned public repository (not yet created)')
    try:
        head = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        head = 'nogit'
    o.put('meta.code_git', head, 'derived: git rev-parse --short HEAD at export time')
    files = sorted(p.relative_to(RES).as_posix() for p in RES.rglob('*') if p.is_file() and not p.name.startswith('.'))
    o.put('meta.n_result_files', len(files), 'derived: count of files under results/')
    prov = {}
    for f in files:
        if f.endswith('.json'):
            j = json.loads((RES / f).read_text())
            g = j.get('git') or (j.get('provenance') or {}).get('git')
            if g:
                prov[f] = g
    o.put('meta.result_git', prov, 'derived: the "git" (or provenance.git) stamp of each results/*.json')
    added = {}
    for f in prov:
        try:
            h = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%h', '--', f'results/{f}'], cwd=ROOT, text=True).split()
        except Exception:
            h = []
        added[f] = h[-1] if h else None
    o.put('meta.result_commit', added, 'derived: git log --diff-filter=A (commit that first added each results/*.json; null = not committed yet)')

    # ---- KT1 (H2)
    k1 = load('kt1.json')
    for k in ['D_star_deg', 'kappa_v_best', 'posthoc_D_star_excl_ratio1_deg', 'same_kappa_maxdev_deg', 'vecsum_maxdev_deg', 'ring_converged_maxchange_deg']:
        o.copy(f'kt1.{k}', 'kt1.json', f'primary/{k}', k1)
    for k in ['kappa', 'g_obj', 'tau_s', 'ring_gain']:
        o.copy(f'kt1.ring_params.{k}', 'kt1.json', f'primary/ring_params/{k}', k1)
    o.copy('kt1.gate', 'kt1.json', 'gate', k1)
    o.copy('kt1.git', 'kt1.json', 'provenance/git', k1)
    cells = k1['primary']['cells']
    for k in ['delta_deg', 'ratio', 'ring_deg', 'vote_best_deg', 'vecsum_deg', 'valid']:
        o.put(f'kt1.cells.{k}', [c[k] for c in cells], f'results/kt1.json:primary/cells/*/{k}')
    sens = [s['D_star_deg'] for s in k1['sensitivity']]
    o.put('kt1.sensitivity_range', [min(sens), max(sens)], 'derived: min, max of results/kt1.json:sensitivity/*/D_star_deg')
    o.put('kt1.n_sensitivity', len(sens), 'derived: len(results/kt1.json:sensitivity)')
    for dl in k1['hysteresis']:
        for arm in ['ring', 'vote_same_kappa', 'vote']:
            for k in ['ratios', 'heading_deg', 'switch_up_ratio', 'switch_back_ratio']:
                o.copy(f'kt1.hysteresis.d{dl}.{arm}.{k}', 'kt1.json', f'hysteresis/{dl}/{arm}/{k}', k1)
    for k in ['D_star_deg', 'vecsum_maxdev_deg', 'posthoc_D_star_excl_ratio1_deg']:
        o.copy(f'kt1.tuned.{k}', 'kt1.json', f'secondary_tuned_ring/{k}', k1)

    # ---- KT2 (H1 pilot)
    k2, tp = load('kt2.json'), load('kt2_tuned_params.json')
    h1_block(o, 'kt2', 'kt2.json', k2, 'kt2_test.parquet')
    for a in ARMS:
        o.copy(f'kt2.train_auc.{a}', 'kt2_tuned_params.json', f'arms/{a}/train_auc', tp)
    o.copy('kt2.tune_cpu_s', 'kt2_tuned_params.json', 'cpu_s', tp)
    o.copy('kt2.tune_wall_s', 'kt2_tuned_params.json', 'wall_s', tp)
    o.put('kt2.n_configs', len(tp['arms']['ring']['rung1_auc']), 'derived: len(results/kt2_tuned_params.json:arms/ring/rung1_auc)')
    o.put('kt2.n_rung2', len(tp['arms']['ring']['rung2']), 'derived: len(results/kt2_tuned_params.json:arms/ring/rung2)')

    # ---- KT3 (H3, 2D)
    k3 = load('kt3.json')
    h3_block(o, 'kt3', 'kt3.json', k3)

    # ---- H4 recognition
    ra, rg, rc, rd = load('recog_auroc.json'), load('recog_gate.json'), load('recog_cost.json'), load('recog_diag.json')
    o.copy('h4.thetas', 'recog_auroc.json', 'thetas', ra)
    o.put('h4.n_layouts', len(ra['layouts']), 'derived: len(results/recog_auroc.json:layouts)')
    o.copy('h4.layouts', 'recog_auroc.json', 'layouts', ra)
    o.copy('h4.git', 'recog_auroc.json', 'git', ra)
    for r in RECOG:
        for lf in ['lf_off', 'lf_on']:
            src = f'recognisers/{r}/{lf}'
            for k, kk in [('auroc', 'auroc'), ('lo', 'ci95_lo'), ('hi', 'ci95_hi'), ('catchment', 'catchment_bins'),
                          ('catchment_ci', 'catchment_bins_ci95'), ('catchment08', 'catchment_bins_at_0.8'),
                          ('mean_out', 'mean_auroc_out_of_arc'), ('mean_all', 'mean_auroc_all'),
                          ('n_pos', 'n_pos_per_angle'), ('n_neg', 'n_neg_per_angle')]:
                o.copy(f'h4.rec.{r}.{lf}.{k}', 'recog_auroc.json', f'{src}/{kk}', ra)
        for k, kk in [('gain', 'lf_gain_out_of_arc_mean_auroc'), ('gain_ci', 'ci95'), ('widening', 'widening_ratio'),
                      ('catch_off_on', 'catchment_bins_off_on')]:
            o.copy(f'h4.paired.{r}.{k}', 'recog_auroc.json', f'paired/{r}/{kk}', ra)
    for lf in ['lf0', 'lf1']:
        o.copy(f'h4.mb_minus_pixels.{lf}', 'recog_auroc.json', f'paired/mb_minus_pixels_{lf}/by_angle', ra)
    for g, ks in {'a_mb_below_0.75_at_0deg': ['fires', 'value'],
                  'b_mb_not_above_pixels': ['fires', 'max_mb_minus_pixels_lf_on', 'max_mb_minus_pixels_lf_off', 'fires_lf_off_version'],
                  'c_lf_widening_below_1.5': ['fires', 'value', 'catchment_off', 'catchment_on'],
                  'd_colour_ge_mb': ['fires', 'colour_lf_on', 'mb_lf_on', 'colour_lf_off', 'mb_lf_off']}.items():
        for k in ks:
            o.copy(f'h4.gates.{g[0]}.{k}', 'recog_gate.json', f'gates/{g}/{k}', rg)
    o.copy('h4.claim', 'recog_gate.json', 'learning_flight_claim', rg)
    for r in ['mb', 'pixels', 'colour', 'dino_patch', 'dino_cls']:
        for lf in ['lf_off', 'lf_on']:
            o.copy(f'h4.cost.bytes.{lf}.{r}', 'recog_cost.json', f'stored_goal_bytes/{lf}/{r}', rc)
        o.copy(f'h4.cost.model_bytes.{r}', 'recog_cost.json', f'model_bytes/{r}', rc)
        o.copy(f'h4.cost.ms.{r}', 'recog_cost.json', f'ms_per_query_median/{r}', rc)
    o.put('h4.cost.dino_params', rc['model_bytes']['dino_patch'] // 4, 'derived: results/recog_cost.json:model_bytes/dino_patch / 4 (fp32)')
    o.put('h4.diag.jitter', [0, 0.25, 0.5, 1], 'results/recog_diag.json:results/mb/memory_sun (keys)')
    o.put('h4.diag.n_layouts', len(rd['seed']), 'derived: len(results/recog_diag.json:seed)')
    for r in ['mb', 'pixels', 'colour', 'dino_patch', 'mb_crop', 'pixels_crop']:
        for sun in ['memory_sun', 'new_sun']:
            for j, key in enumerate(['0', '0.25', '0.5', '1']):
                o.copy(f'h4.diag.{r}.{sun}.{j}', 'recog_diag.json', f'results/{r}/{sun}/{key}/auroc', rd)

    # ---- vehicle, parity, surrogate
    vp = load('vehicle_params.json')
    for k, v in vp['params'].items():
        for f in ['value', 'unit', 'kind', 'source']:
            o.copy(f'vehicle.params.{k}.{f}', 'vehicle_params.json', f'params/{k}/{f}', vp)
    for k in ['W20_kt/light', 'W20_kt/moderate', 'W20_kt/severe']:
        o.copy('vehicle.dryden.' + k.split('/')[1], 'vehicle_params.json', 'dryden/' + k, vp)
    qs = load('quad_selfcheck.json')
    for k in ['hover_10s_max_pos_err_m', 'steps/vel_x_1mps_settle_s', 'steps/alt_0p5m_settle_s', 'steps/yaw_1rad_settle_s',
              'gust/severe_mean3_max_tilt_deg', 'gust/severe_mean3_max_horiz_dev_m', 'gust/motor_saturation_frac',
              'params_derived/hover_cmd', 'params_derived/hover_thrust_per_motor_N']:
        o.copy('vehicle.selfcheck.' + k.split('/')[-1], 'quad_selfcheck.json', k, qs)
    pa = load('parity.json')
    for k in ['pass', 'n_scenarios', 'max_pos_err_m', 'tolerance_m', 'duration_s', 'dt_s']:
        o.copy(f'parity.quad.{k}', 'parity.json', k, pa)
    o.copy('parity.quad.git', 'parity.json', 'provenance/git', pa)
    mj = load('quad_mujoco_check.json')
    for k in ['max_pos_dev_m', 'duration_s', 'mujoco', 'drag_off/max_pos_dev_m']:
        o.copy('parity.mujoco.' + k.replace('/', '_'), 'quad_mujoco_check.json', k, mj)
    o.put('parity.mujoco.n_cases', len(mj['cases']), 'derived: len(results/quad_mujoco_check.json:cases)')
    ys = load('yaw_surrogate.json')
    for k in ['K', 'tau_s', 'fc_hz', 'rms_gain_err_db', 'max_abs_gain_err_db', 'rms_phase_err_deg', 'max_abs_phase_err_deg', 'js_parity_max_abs_err']:
        o.copy(f'parity.yaw.{k}', 'yaw_surrogate.json', k, ys)
    o.put('parity.yaw.n_freqs', len(ys['per_freq']['f_hz']), 'derived: len(results/yaw_surrogate.json:per_freq/f_hz)')
    pb = load('parity_brain.json')
    if pb:
        o.copy('parity.brain.pass', 'parity_brain.json', 'pass', pb)
        ms = pb['results']['missions']
        o.put('parity.brain.n_missions', len(ms), 'derived: len(results/parity_brain.json:results/missions)')
        o.put('parity.brain.max_pos_err_first_20s_m', max(m['max_pos_err_first_20s_m'] for m in ms),
              'derived: max of results/parity_brain.json:results/missions/*/max_pos_err_first_20s_m')
        o.put('parity.brain.n_modules', len(pb['results']) - 1, 'derived: len(results/parity_brain.json:results) - 1 (missions)')
    else:
        o.put('parity.brain', None, 'derived: results/parity_brain.json missing (pending)')

    # ---- 3D mission (optional)
    m1, m3 = load('mission_h1.json'), load('mission_h3.json')
    if m1:
        h1_block(o, 'mission.h1', 'mission_h1.json', m1, 'mission_test.parquet')
    else:
        o.put('mission.h1', None, 'derived: results/mission_h1.json missing (pending)')
    if m3:
        h3_block(o, 'mission.h3', 'mission_h3.json', m3)
    else:
        o.put('mission.h3', None, 'derived: results/mission_h3.json missing (pending)')

    # ---- the Results tab's chart keys
    o.d = listify(o.d)
    h1pre = 'mission.h1' if m1 and 'aggregate_curve' in o.d['mission']['h1'] else 'kt2'
    h3pre = 'mission.h3' if m3 else 'kt3'
    app_keys(o, h1pre, h3pre, 'results/mission_h1.json (3D)' if h1pre != 'kt2' else 'results/kt2.json (2D pilot)',
             'results/mission_h3.json (3D)' if m3 else 'results/kt3.json (2D)')
    return o


# ---------------------------------------------------------------- self-check
def leaves(x, p=''):
    if isinstance(x, dict) and x:
        for k, v in x.items():
            yield from leaves(v, f'{p}.{k}' if p else k)
    elif isinstance(x, list) and x:
        for i, v in enumerate(x):
            yield from leaves(v, f'{p}.{i}')
    else:
        yield p, x


def selfcheck(doc):
    d, src = doc, doc['_source']
    n_file = 0
    # 1. every copied value equals its source
    for path, s in src.items():
        m = re.match(r'^results/([^:]+):(.+)$', s)
        if not m or ' (' in s or '*' in s:
            continue
        val = _clean(get(json.loads((RES / m[1]).read_text()), m[2]))
        assert get(d, path, '.') == val, (path, s)
        n_file += 1
    # 2. wildcard copies
    k1 = load('kt1.json')
    for k in ['delta_deg', 'ratio', 'ring_deg', 'vote_best_deg', 'vecsum_deg', 'valid']:
        assert d['kt1']['cells'][k] == _clean([c[k] for c in k1['primary']['cells']])
    # 3. derived values re-derived independently
    runs = pd.read_parquet(RES / 'runs' / 'kt2_test.parquet')
    runs['succ'] = runs.success_target & runs.success_home
    k2 = load('kt2.json')
    n = d['kt2']['n_seeds']
    for a in ARMS:
        r = runs[runs.arbiter == a]
        for f in d['kt2']['faults']:
            g = r[r.fault == f].groupby('level').succ
            cnt, tot = g.sum().tolist(), g.size().tolist()
            assert all(t == n for t in tot)
            for i, c in enumerate(d['kt2']['curves'][a][f]):
                assert abs(c['success'] - cnt[i] / n) < 1e-12, (a, f, i)
                ph = cnt[i] / n
                lo = (ph + Z**2 / (2 * n) - Z * math.sqrt(ph * (1 - ph) / n + Z**2 / (4 * n * n))) / (1 + Z**2 / n)
                assert abs(c['ci'][0] - max(0, lo)) < 1e-12 and c['ci'][0] <= c['success'] <= c['ci'][1]
        agg = d['h1_success_vs_severity'][a] if d['h1_success_vs_severity_from'].startswith('results/kt2') else None
        if agg:
            assert abs(agg[0]['success'] - r[r.fault == 'none'].succ.mean()) < 1e-12        # x = 0 is the no-fault run
            last = np.mean([r[(r.fault == f) & (r.level == r[r.fault == f].level.max())].succ.mean() for f in d['kt2']['faults']])
            assert abs(agg[-1]['success'] - last) < 1e-12                                     # x = 1 is every fault's worst level
            assert all(q['ci'][0] <= q['success'] <= q['ci'][1] for q in agg)
            # the curve's area is the pre-registered aggregate AUC (trapezoid per fault), up to the grid for 3-level gust
            auc = np.trapezoid([q['success'] for q in agg], GRID)
            assert abs(auc - k2['aggregate_auc'][a]['mean']) < 0.01, (a, auc)
    for a in ARMS:
        for p, q in zip(d['kt3']['points'][a], load('kt3.json')['points'][a]):
            assert abs(p['hold_ci'][0] - (q['hold_mean_s'] - Z * q['hold_sem'])) < 1e-12
            assert abs(p['release_ci'][1] - (q['release_mean_s'] + Z * q['release_sem'])) < 1e-12
    for app, fk in APP_RECOG.items():
        assert d['recog_auroc'][app]['no_flight']['mean'] == load('recog_auroc.json')['recognisers'][fk]['lf_off']['auroc'][0]
    assert d['h4']['cost']['dino_params'] == 22056576                                        # DINOv2 ViT-S/14
    assert d['kt1']['sensitivity_range'][0] == min(s['D_star_deg'] for s in k1['sensitivity'])
    # 4. pre-registered verdicts the paper's prose is written around
    assert d['kt1']['gate'] == 'PASS' and d['kt1']['D_star_deg'] > 5
    assert d['kt3']['gate'] == 'DEAD' and d['h4']['claim'] == 'KILLED'
    assert d['h4']['gates']['a']['fires'] and d['h4']['gates']['b']['fires'] and d['h4']['gates']['c']['fires'] and not d['h4']['gates']['d']['fires']
    assert d['parity']['quad']['pass'] is True
    # 5. sources point only at results/ files or derived formulas
    assert all(v.startswith('results/') or v.startswith('derived:') for v in src.values())
    # 5. every leaf has a source; the app's chart keys have the shapes panels.js reads
    for p, _ in leaves({k: v for k, v in d.items() if k != '_source'}):
        assert any(p == k or p.startswith(k + '.') for k in src), f'no _source for {p}'
    for a in ARMS:
        for p in d['stubbornness_frontier'][a]:
            assert set(p) == {'gain_or_knob', 'hold_s', 'hold_ci', 'release_s', 'release_ci'} and len(p['hold_ci']) == 2
        for q in d['h1_success_vs_severity'][a]:
            assert 0 <= q['severity'] <= 1 and 0 <= q['success'] <= 1 and len(q['ci']) == 2
    for m in APP_RECOG:
        assert set(d['recog_auroc'][m]) == {'no_flight', 'flight'}
    s = (OUT / 'summary.js').read_text()
    assert json.loads(s[s.index('=') + 1:].strip().rstrip(';')) == json.loads((OUT / 'summary.json').read_text())
    return n_file


def main():
    o = build()
    doc = dict(o.d, _source=o.src)
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)
    (OUT / 'summary.json').write_text(txt + '\n')
    (OUT / 'summary.js').write_text('// Generated by scripts/export_app_summary.py; do not edit.\nwindow.APP_SUMMARY = ' + txt + ';\n')
    n_file = selfcheck(json.loads((OUT / 'summary.json').read_text()))
    pend = [k for k, v in [('mission_h1', doc['mission']['h1']), ('mission_h3', doc['mission']['h3']), ('parity_brain', doc['parity']['brain'])] if v is None]
    print(f'wrote app/data/summary.json (+ summary.js): {len(o.src)} sources, {n_file} re-read from result files; '
          f'pending: {", ".join(pend) or "none"}')
    print('self-check passed')


if __name__ == '__main__':
    main()
