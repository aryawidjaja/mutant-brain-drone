"""Post hoc diagnostics of the 3D mission results (labelled post hoc; no new simulation, no change to any
pre-registered number). Reads results/runs/mission_h3.parquet and results/runs/mission_test.parquet.

H3 per sweep point (both protocols): target-stop rate, share of runs that hit the time limit, share of decoy runs
that stopped at neither object, share of occlusion runs whose steering was not on the target when the occlusion
began (hold < 0.05 s), and the hold restricted to runs that were on the target at onset.
H1: the arm-permutation test with each arm in turn as the target (is the 3D H1 gap specific to the ring?), and the
aggregate AUC without the swapped-decoy fault.
-> results/mission_diag.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.arbiters import ARBITERS
from kt3_stubbornness import SWEEPS
from provenance import stamp
from run_3d import FAULT_SEV, auc_table
from stats import arm_permutation_test

ROOT = Path(__file__).resolve().parents[1]
T_LIMIT = 300.0
ON_TARGET_EPS = 0.05


def h3_points(d):
    d = d.assign(pt=d.unit.str.split("__").str[1].str[1:].astype(int), proto=d.unit.str.split("__").str[2])
    out = {}
    for arm, (knob, vals) in SWEEPS.items():
        rows = []
        for j, v in enumerate(vals):
            o = d[(d.arbiter == arm) & (d.pt == j) & (d.proto == "occlusion")]
            c = d[(d.arbiter == arm) & (d.pt == j) & (d.proto == "decoy")]
            held = o.t_hold_occlusion_s
            on = held[held >= ON_TARGET_EPS]
            rows.append({"knob": knob, "value": None if v is None else float(v),
                         "occlusion_target_stop_rate": float(o.success_target.mean()),
                         "occlusion_time_limit_rate": float((o.time_s >= T_LIMIT - 1e-6).mean()),
                         "occlusion_not_on_target_at_onset": float((held < ON_TARGET_EPS).sum() / held.notna().sum()),
                         "hold_on_target_at_onset_mean_s": float(on.mean()) if len(on) else None, "n_on_target_at_onset": int(len(on)),
                         "hold_max_s": float(held.max()),
                         "decoy_target_stop_rate": float(c.success_target.mean()),
                         "decoy_wrong_stop_rate": float(c.wrong_object_stop.mean()),
                         "decoy_stopped_at_neither": float((~c.success_target & ~c.wrong_object_stop).mean()),
                         "decoy_time_limit_rate": float((c.time_s >= T_LIMIT - 1e-6).mean())})
        out[arm] = rows
    return out


def h1_arms(d):
    d = d.copy()
    d.loc[d.fault == "none", "severity"] = np.nan
    # severity in the test parquet is the parameter value as a string; rebuild the severity fraction from 'level'
    frac = {f: dict(enumerate(ss, start=1)) for f, ss in FAULT_SEV.items()}
    d["severity"] = [np.nan if f == "none" else frac[f][int(l)] for f, l in zip(d.fault, d.level)]
    auc = {a: auc_table(d[d.arbiter == a]) for a in ARBITERS}
    agg = {a: auc[a].mean(1).sort_index().to_numpy() for a in ARBITERS}
    no_decoy = {a: float(auc[a].drop(columns="decoy").mean(1).mean()) for a in ARBITERS}
    perm = {a: arm_permutation_test(agg[a], [agg[b] for b in ARBITERS if b != a], n_perm=5000) for a in ARBITERS}
    return {"aggregate_auc": {a: float(agg[a].mean()) for a in ARBITERS}, "aggregate_auc_without_decoy_fault": no_decoy,
            "each_arm_minus_best_other_permutation": perm}


def main():
    h3 = h3_points(pd.read_parquet(ROOT / "results/runs/mission_h3.parquet"))
    h1 = h1_arms(pd.read_parquet(ROOT / "results/runs/mission_test.parquet"))
    stored = json.loads((ROOT / "results/mission_h1.json").read_text())["aggregate_auc"]
    for a in ARBITERS:                                        # the rebuilt AUCs must equal the reported ones
        assert abs(h1["aggregate_auc"][a] - stored[a]["mean"]) < 1e-12, a
    res = {"label": "post hoc diagnostics (not pre-registered)", "h3": h3, "h1": h1,
           "definitions": {"time_limit_s": T_LIMIT, "not_on_target_at_onset": f"hold < {ON_TARGET_EPS} s",
                           "stopped_at_neither": "neither a target stop nor a wrong-object stop"},
           "provenance": stamp(seed="test 1000-1299", param_source="results/runs/mission_h3.parquet, results/runs/mission_test.parquet")}
    (ROOT / "results/mission_diag.json").write_text(json.dumps(res, indent=1))
    for g in (0, 3, 6, 8, 9, 10, 13, 15):
        p = h3["ring"][g]
        print(f"ring gain {p['value']:4}: occl stop {p['occlusion_target_stop_rate']:.2f} limit {p['occlusion_time_limit_rate']:.2f} "
              f"off-target {p['occlusion_not_on_target_at_onset']:.2f} hold|on {p['hold_on_target_at_onset_mean_s']} | decoy neither {p['decoy_stopped_at_neither']:.2f}")
    for a, t in h1["each_arm_minus_best_other_permutation"].items():
        print(f"{a:14s} minus best other {t['est']:+.4f}  p {t['p_two_sided']:.3f}  no-decoy AUC {h1['aggregate_auc_without_decoy_fault'][a]:.4f}")
    return res


if __name__ == "__main__":
    r = main()
    assert set(r["h3"]) == set(SWEEPS) and len(r["h3"]["ring"]) == len(SWEEPS["ring"][1])
    assert all(0 <= p["occlusion_target_stop_rate"] <= 1 for ps in r["h3"].values() for p in ps)
    print("mission diagnostics OK")
