"""Showcase replays for the app's Results mode, recorded with the Python reference (scripts/mission.py).

Six illustrative missions on test seeds (forest, obstacle density 0.3, the 3D study's world): the ring and the state
machine on the same swapped-decoy seed, the ring at high and at zero recurrent gain on the same occlusion seed, a
no-fault success and a false-sun case. Seeds were picked to show contrasting behaviour; they illustrate the
mechanisms and are not results (the paper's numbers come from all test seeds).
Arbiter parameters: results/tuned3d_params.json when present, else results/kt2_tuned_params.json.

Writes app/runs/<name>.json ({meta, world, frames}), app/runs/manifest.json and app/runs/bundle.js
(window.RUN_BUNDLE = {manifest, runs}). Frames are the mission's C6 frames thinned to 5 Hz with numbers rounded to
3 significant digits (time and outcome exact); the eye panorama is kept every 2 s as base64 bytes. Each frame's pi also carries
heading_est_deg (the drone's compass heading at that tick) so the app can draw the home vector in the world frame.
Usage: uv run python scripts/export_runs.py
"""
import base64, json, math, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "brain")]
from mission import Mission  # noqa: E402
from provenance import stamp  # noqa: E402
from run_3d import scen  # noqa: E402

OUT = ROOT / "app/runs"
FRAME_EVERY, EYE_EVERY = 2, 20          # of the mission's 10 Hz frames: 5 Hz poses, eye every 2 s
SHOW = [  # name, arm, seed, fault, severity, ring_gain override, label
    ("decoy_ring_1010", "ring", 1010, "decoy", 2 / 3, None, "Swapped decoy (similarity 0.8), ring hub: the look-alike on the reported spot"),
    ("decoy_sm_1010", "state_machine", 1010, "decoy", 2 / 3, None, "Swapped decoy (similarity 0.8), state machine: same seed as the ring run"),
    ("occl_ring_g2_1003", "ring", 1003, "occlusion", 1.0, 2.0, "Target hidden for 10 s, ring gain 2.0 (strong recurrence)"),
    ("occl_ring_g0_1003", "ring", 1003, "occlusion", 1.0, 0.0, "Target hidden for 10 s, ring gain 0 (no recurrence); same seed"),
    ("none_ring_1004", "ring", 1004, "none", None, None, "No fault, ring hub: find the tent and fly home"),
    ("sun_ring_1002", "ring", 1002, "false_sun", 2 / 3, None, "False sun (90° compass offset during the outbound flight), ring hub"),
]


def r4(v):
    if isinstance(v, float):
        return v if not math.isfinite(v) or v == 0 else float(f"{v:.3g}")
    if isinstance(v, dict):
        return {k: (x if k in ("t", "outcome") else r4(x)) for k, x in v.items()}
    if isinstance(v, list):
        return [r4(x) for x in v]
    return v


def record(arm, seed, fault, sev, gain, tuned):
    p = dict(tuned[arm]["params"])
    if gain is not None:
        p["ring_gain"] = gain
    sc = scen(arm, p, seed, "test", fault, sev)
    m = Mission([sc], log_rows=[0])
    n = 0
    T = sc["mission"]["max_time_s"]
    while m.t < T - 1e-9 and m.phase[0] < 4:
        m.step()
        if len(m.frames[0]) > n:                       # a frame was logged during this tick
            m.frames[0][-1]["pi"]["heading_est_deg"] = math.degrees(float(m.heading_est[0]))
            n = len(m.frames[0])
    out = m.outcome_row(0)
    fr = m.frames[0]
    keep = [f for i, f in enumerate(fr) if i % FRAME_EVERY == 0 or i == len(fr) - 1]
    for i, f in enumerate(keep):
        e = f.pop("eye")
        if i % (EYE_EVERY // FRAME_EVERY) == 0:
            f["eye"] = {"w": e["w"], "h": e["h"], "b64": base64.b64encode(bytes(e["rgb"])).decode()}
    keep[-1]["outcome"] = out
    meta_sc = {k: v for k, v in sc.items() if not k.startswith("_")}
    return meta_sc, m.worlds[0], [r4(f) for f in keep], out


def main():
    src = next(p for p in ("results/tuned3d_params.json", "results/kt2_tuned_params.json") if (ROOT / p).exists())
    tuned = json.loads((ROOT / src).read_text())["arms"]
    st = stamp(param_source=src)
    OUT.mkdir(exist_ok=True)
    man, runs = [], {}
    for name, arm, seed, fault, sev, gain, label in SHOW:
        t0 = time.time()
        sc, world, frames, out = record(arm, seed, fault, sev, gain, tuned)
        run = {"meta": {"scenario": sc, "arbiter": arm, "seed": seed, "git": st["git"], "sim_version": st["sim_version"],
                        "created": st["created"], "label": label, "params_source": src, "outcome": out,
                        "note": "Illustrative replay recorded by scripts/export_runs.py with the Python reference; not a result."},
               "world": world, "frames": frames}
        f = f"{name}.json"
        (OUT / f).write_text(json.dumps(run, separators=(",", ":")))
        man.append({"file": f, "label": f"{label} · seed {seed}"})
        runs[f] = run
        print(f"{name}: {len(frames)} frames, t_end {out['time_s']:.1f} s, target {out['success_target']}, home {out['success_home']}, "
              f"decoy stop {out['wrong_object_stop']}, {(OUT / f).stat().st_size / 1e6:.2f} MB, {time.time() - t0:.0f} s")
    manifest = {"runs": man, "_source": "scripts/export_runs.py", **{k: st[k] for k in ("git", "sim_version", "created")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "bundle.js").write_text("window.RUN_BUNDLE = " + json.dumps({"manifest": manifest, "runs": runs}, separators=(",", ":")) + ";\n")
    return manifest, runs


if __name__ == "__main__":
    manifest, runs = main()
    # self-check: files round-trip, every run has frames in time order with the C6 fields the app reads, the eye
    # decodes to w*h*3 bytes, the last frame carries the outcome, and the bundle stays modest
    total = sum((OUT / r["file"]).stat().st_size for r in manifest["runs"]) + (OUT / "bundle.js").stat().st_size
    assert len(manifest["runs"]) == len(SHOW) and (OUT / "bundle.js").stat().st_size < 8e6, total
    for r in manifest["runs"]:
        run = json.loads((OUT / r["file"]).read_text())
        fr = run["frames"]
        assert run["meta"]["seed"] >= 1000 and run["world"]["objects"] and len(fr) > 10
        assert all(b["t"] > a["t"] for a, b in zip(fr, fr[1:]))
        assert all({"quad", "pi", "ring", "arbiter", "objects", "mind", "phase"} <= set(f) for f in fr)
        assert all(math.isfinite(f["pi"]["heading_est_deg"]) for f in fr)
        e = next(f["eye"] for f in fr if "eye" in f)
        assert len(base64.b64decode(e["b64"])) == e["w"] * e["h"] * 3
        assert fr[-1]["outcome"] == run["meta"]["outcome"]
    o = {r["file"]: json.loads((OUT / r["file"]).read_text())["meta"]["outcome"] for r in manifest["runs"]}
    assert o["decoy_ring_1010.json"]["wrong_object_stop"] != o["decoy_sm_1010.json"]["wrong_object_stop"], "decoy pair no longer contrasts"
    print(f"export_runs self-check OK; {len(manifest['runs'])} runs, {total / 1e6:.1f} MB on disk (runs + bundle)")
