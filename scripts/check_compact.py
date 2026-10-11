"""Check that dropping finished rows mid-batch (Mission.run(compact=True)) leaves every outcome unchanged:
discrete fields identical, continuous fields within 1e-9 (summation order only).
Usage: uv run python scripts/check_compact.py ring state_machine accumulator vote sum ring_static  (~6 min)"""
import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from scenario import sample_scenario
from mission import Mission
from brain.arbiters import ARBITERS
from kt2_five_arms import N_CONFIGS
import run_3d as r
faults = [("none", None), ("occlusion", 1.0), ("decoy", 2/3), ("compass_half", 1.0), ("eye_columns", 1.0),
          ("false_sun", 1.0), ("speed_scale", 1.0), ("gust", 0.9), ("gust", 0.1)]
bad = 0
for arm in sys.argv[1:]:
    p = dict(r.configs(arm)[5]); g = float(p.pop("ring_gain", 0.0)) if arm == "ring" else 0.0
    if arm == "ring": p["g_home"] = 3.0
    scs = [sample_scenario(s, fault=f, severity=v, arbiter=arm, ring_gain=g, params=p, landscape="forest",
                           obstacle_density=0.3, dist_m=(6.0, 14.0), max_time_s=float(40 + 10 * (s % 5)))
           for s in range(18) for f, v in [faults[s % len(faults)]]]
    t0 = time.time(); a = Mission(scs).run(compact=False); t1 = time.time(); b = Mission(scs).run(compact=True); t2 = time.time()
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        same = np.array_equal(x, y, equal_nan=True) if a[k].dtype.kind in "bi" or k.startswith("t_") or k == "time_s" \
            else (np.array_equal(np.isnan(x), np.isnan(y)) and np.nanmax(np.abs(x - y), initial=0.0) <= 1e-9)
        bad += not same
        if not same: print("  DIFF", arm, k, "max abs", float(np.nanmax(np.abs(x - y))), "nan-mismatch", int((np.isnan(x) != np.isnan(y)).sum()))
    print(f"{arm}: plain {t1-t0:.0f}s compact {t2-t1:.0f}s; ended at {np.round(a['time_s'],1).tolist()[:8]}... success_target {a['success_target'].sum()}")
assert bad == 0, f"{bad} fields differ"
print("compaction check OK: discrete outcomes identical, continuous within 1e-9")
