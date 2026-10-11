"""Vehicle parity: Python reference (scripts/quad.py + dryden.py, batched) vs browser port (app/sim/quad.js +
dryden.js, batch 1, run in Node via scripts/parity_runner.mjs) on fixed-seed scenarios, 10 s each.

PLAN gate: max position error < 1 cm over 10 s in every scenario. Writes results/parity.json; exits 1 on fail.
Brain-module parity lives in scripts/parity_brain.py (workstream D) -> results/parity_brain.json; it can reuse
run_node() below.
Run: uv run python scripts/parity_check.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dryden import Dryden
from provenance import stamp
from quad import Controller, Quad

ROOT = Path(__file__).resolve().parents[1]
DT, T_S, EVERY, TOL_M = 0.002, 10.0, 10, 0.01


def sp(t, vel=(0, 0), alt=1.0, yaw=0.0):
    return {"t": t, "vel_xy": list(vel), "alt": alt, "yaw": yaw}


def scen(name, schedule, seed=0, turbulence="none", mean_wind=(0, 0, 0), pos0=(0, 0, 1.0), yaw0=0.0):
    return {"name": name, "schedule": schedule, "seed": seed, "turbulence": turbulence, "mean_wind": list(mean_wind),
            "pos0": list(pos0), "yaw0": yaw0, "dryden_alt_m": 3.0}


SCENARIOS = [
    scen("hover", [sp(0)]),
    scen("takeoff_from_ground", [sp(0, alt=2.0)], pos0=(0, 0, 0)),
    scen("vel_step_x", [sp(0), sp(1, vel=(1.5, 0))]),
    scen("vel_step_diag_and_stop", [sp(0), sp(1, vel=(1, -1)), sp(5, vel=(0, 0))]),
    scen("yaw_steps_wrap", [sp(0), sp(1, yaw=1.5), sp(5, yaw=-2.5)], yaw0=0.3),
    scen("alt_step", [sp(0), sp(1, alt=2.5), sp(6, alt=0.5)]),
    scen("combined_manoeuvre", [sp(0), sp(1, vel=(2, 0.5), alt=2, yaw=0.8), sp(6, vel=(-1, 1), alt=1.5, yaw=-0.5)]),
    scen("dryden_light", [sp(0, alt=3)], seed=1001, turbulence="light", pos0=(0, 0, 3)),
    scen("dryden_moderate_vel", [sp(0, alt=3), sp(2, vel=(1, 0), alt=3)], seed=1002, turbulence="moderate", pos0=(0, 0, 3)),
    scen("dryden_severe", [sp(0, alt=3)], seed=1003, turbulence="severe", pos0=(0, 0, 3)),
    scen("mean_wind_only", [sp(0, alt=2)], mean_wind=(2, -1, 0), pos0=(0, 0, 2)),
    scen("severe_plus_mean_yawed", [sp(0, alt=3), sp(3, vel=(0.5, 0.5), alt=3, yaw=2.0)], seed=7, turbulence="severe",
         mean_wind=(3, 0, 0), pos0=(5, -3, 3), yaw0=0.7),
]


def run_node(runner, payload):
    """Run a Node script with JSON on stdin; return its parsed JSON stdout."""
    r = subprocess.run(["node", str(runner)], input=json.dumps(payload), capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(r.stdout)


def run_python(scenarios):
    """All scenarios as one batch (one Dryden per row, since intensity differs)."""
    B = len(scenarios)
    q = Quad(B, pos0=np.array([s["pos0"] for s in scenarios]), yaw0=np.array([s["yaw0"] for s in scenarios]))
    c = Controller(B)
    dry = [Dryden(1, [s["seed"]], s["turbulence"], s["mean_wind"], alt_m=s["dryden_alt_m"]) for s in scenarios]
    log = {"pos": [], "euler": [], "wind": []}
    for k in range(int(round(T_S / DT))):
        t = k * DT
        cur = [[e for e in s["schedule"] if e["t"] <= t + 1e-9][-1] for s in scenarios]
        spb = {"vel_xy": np.array([e["vel_xy"] for e in cur], float), "alt": np.array([e["alt"] for e in cur], float),
               "yaw": np.array([e["yaw"] for e in cur], float)}
        wind = np.vstack([d.step(DT) for d in dry])
        q.step(c(q, spb, DT), wind, DT)
        if (k + 1) % EVERY == 0:
            log["pos"].append(q.pos.copy()); log["euler"].append(q.euler()); log["wind"].append(wind)
    return {k: np.array(v) for k, v in log.items()}  # [N, B, 3]


def main():
    py = run_python(SCENARIOS)
    js = run_node(ROOT / "scripts" / "parity_runner.mjs", {"dt": DT, "T": T_S, "every": EVERY, "scenarios": SCENARIOS})
    rows = []
    for b, s in enumerate(SCENARIOS):
        j = {k: np.array(v) for k, v in js[s["name"]].items()}
        assert j["pos"].shape == py["pos"][:, b].shape
        dpos = np.linalg.norm(j["pos"] - py["pos"][:, b], axis=1)
        rows.append({"name": s["name"], "turbulence": s["turbulence"], "seed": s["seed"],
                     "max_pos_err_m": float(dpos.max()),
                     "max_att_err_rad": float(np.abs(np.angle(np.exp(1j * (j["euler"] - py["euler"][:, b])))).max()),
                     "max_wind_err_mps": float(np.abs(j["wind"] - py["wind"][:, b]).max()),
                     "path_len_m": float(np.linalg.norm(np.diff(py["pos"][:, b], axis=0), axis=1).sum()),
                     "final_pos_py": py["pos"][-1, b].tolist()})
    worst = max(r["max_pos_err_m"] for r in rows)
    ok = worst < TOL_M and len(rows) >= 10
    res = {"what": "vehicle parity: scripts/quad.py+dryden.py (batched) vs app/sim/quad.js+dryden.js (Node, batch 1)",
           "dt_s": DT, "duration_s": T_S, "sample_every_steps": EVERY, "tolerance_m": TOL_M,
           "n_scenarios": len(rows), "max_pos_err_m": worst, "pass": bool(ok), "scenarios": rows,
           "scenario_defs": SCENARIOS,
           "provenance": stamp(seed=[s["seed"] for s in SCENARIOS], param_source="results/vehicle_params.json",
                               node=subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip())}
    (ROOT / "results" / "parity.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['name']:28s} pos {r['max_pos_err_m']:.2e} m  att {r['max_att_err_rad']:.2e} rad  wind {r['max_wind_err_mps']:.2e}  path {r['path_len_m']:.1f} m")
    print(f"max_pos_err_m = {worst:.3e} (tol {TOL_M}) -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = main()
    # self-check: the comparison is not vacuous (scenarios actually move and the wind actually varies)
    res = json.loads((ROOT / "results" / "parity.json").read_text())
    assert max(r["path_len_m"] for r in res["scenarios"]) > 5.0
    assert np.isfinite(res["max_pos_err_m"])
    sys.exit(0 if ok else 1)
