"""Showcase replays for the app's Results mode, recorded with the Python reference (scripts/mission.py).

Study 2 (v2 brain, the app's default): SHOW_V2, one replay per condition below, all on the FIRST v2 test seed (2000),
with the v2 ring (RingGated) and its frozen parameters (results/tuned_v2_params.json, heading from
results/kt5_tuned_params.json), plus the Study 1 (v1) brain on the same seed for "no fault" and "false sun". The rule
(first test seed, fixed condition list, the study's middle severities) was written here before any of these runs was
recorded, so the replays are not picked by outcome. They illustrate the mechanisms and are not results (the paper's
numbers come from all 300 test seeds).
Study 1 (v1 brain): SHOW, two of the first study's replays kept for contrast (the ring and the state machine on the
same swapped-decoy seed; v1 3D test seed 1010, picked to show contrasting behaviour), arbiter parameters from
results/tuned3d_params.json.

Writes app/runs/<name>.json ({meta, world, frames}), app/runs/manifest.json and app/runs/bundle.js
(window.RUN_BUNDLE = {manifest, runs}). Frames are the mission's C6 frames thinned to 5 Hz with numbers rounded to
3 significant digits (time and outcome exact); the eye panorama is kept every 2 s as base64 bytes. Each frame's pi also carries
heading_est_deg (the drone's compass heading at that tick) so the app can draw the home vector in the world frame.
Usage (each recording is one short process; run them one at a time, then bundle):
  uv run python scripts/export_runs.py record <name>      one replay from SHOW_V2 (or SHOW) -> app/runs/<name>.json
  uv run python scripts/export_runs.py bundle             manifest.json + bundle.js from the files on disk + self-check
  uv run python scripts/export_runs.py v1                 re-record the Study 1 replays (SHOW)
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
V2_EVERY, V2_EYE_EVERY = 3, 36          # Study 2 replays (longer missions): 3.3 Hz poses, eye every 3.6 s, to keep app/runs small
SHOW = [  # Study 1 (v1 brain): name, arm, seed, fault, severity, ring_gain override, label
    ("decoy_ring_1010", "ring", 1010, "decoy", 2 / 3, None, "Study 1 brain · swapped decoy (similarity 0.8), ring hub: the look-alike on the reported spot"),
    ("decoy_sm_1010", "state_machine", 1010, "decoy", 2 / 3, None, "Study 1 brain · swapped decoy (similarity 0.8), state machine: same seed as the ring run"),
]
V2_SEED = 2000                           # the first v2 test seed (fixed rule, see the docstring)
SHOW_V2 = [  # name, brain version, fault, severity (run_3d.FAULT_SEV scale), operator tip ("v1" = default), label
    ("v2_none", 2, "none", None, "v1", "Study 2 brain · no fault"),
    ("v2_sun90", 2, "false_sun", 2 / 3, "v1", "Study 2 brain · false sun 90°"),
    ("v2_compass", 2, "compass_half", 1.0, "v1", "Study 2 brain · half the compass silenced"),
    ("v2_gust", 2, "gust", 0.5, "v1", "Study 2 brain · moderate gusts"),
    ("v2_decoy", 2, "decoy", 2 / 3, "v1", "Study 2 brain · swapped decoy (similarity 0.8)"),
    ("v2_occl7", 2, "occlusion", 2 / 3, "v1", "Study 2 brain · target hidden for 7 s"),
    ("v2_notip", 2, "none", None, None, "Study 2 brain · no operator tip (search from home)"),
    ("v1_none", 1, "none", None, "v1", "Study 1 brain · no fault (same seed as the Study 2 run)"),
    ("v1_sun90", 1, "false_sun", 2 / 3, "v1", "Study 1 brain · false sun 90° (same seed as the Study 2 run)"),
]


def r4(v):
    if isinstance(v, float):
        return v if not math.isfinite(v) or v == 0 else float(f"{v:.3g}")
    if isinstance(v, dict):
        return {k: (x if k in ("t", "outcome") else r4(x)) for k, x in v.items()}
    if isinstance(v, list):
        return [r4(x) for x in v]
    return v


def record(arm, seed, fault, sev, gain, tuned, sc=None, every=FRAME_EVERY, eye_every=EYE_EVERY):
    if sc is None:
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
    keep = [f for i, f in enumerate(fr) if i % every == 0 or i == len(fr) - 1]
    for i, f in enumerate(keep):
        e = f.pop("eye")
        if i % (eye_every // every) == 0:
            f["eye"] = {"w": e["w"], "h": e["h"], "b64": base64.b64encode(bytes(e["rgb"])).decode()}
    keep[-1]["outcome"] = out
    meta_sc = {k: v for k, v in sc.items() if not k.startswith("_")}
    return meta_sc, m.worlds[0], [r4(f) for f in keep], out


def write_run(name, label, arm, seed, sc, world, frames, out, src, st, note):
    run = {"meta": {"scenario": sc, "arbiter": arm, "seed": seed, "git": st["git"], "sim_version": st["sim_version"],
                    "created": st["created"], "label": label, "params_source": src, "outcome": out, "note": note},
           "world": world, "frames": frames}
    (OUT / f"{name}.json").write_text(json.dumps(run, separators=(",", ":")))
    return run


def record_v2(name):
    """One Study 2 replay (SHOW_V2 entry) on V2_SEED: the v2 ring with frozen params, or the v1 brain."""
    from run_v2 import scen2, test_arms
    _, ver, fault, sev, tip, label = next(e for e in SHOW_V2 if e[0] == name)
    arms = test_arms()
    a = arms["v1"] if ver == 1 else arms["v2_ring"]
    kw = {k: v for k, v in a.items() if k not in ("arm", "params")}
    if tip != "v1":
        kw["tip"] = tip
    sc = scen2("ring", a["params"], V2_SEED, "test", fault, sev, **kw)
    src = "results/tuned3d_params.json" if ver == 1 else "results/tuned_v2_params.json + results/kt5_tuned_params.json"
    t0 = time.time()
    meta_sc, world, frames, out = record("ring", V2_SEED, fault, sev, None, None, sc=sc, every=V2_EVERY, eye_every=V2_EYE_EVERY)
    q = lambda x, d: (lambda y: int(y) if y == int(y) else y)(round(x, d)) if isinstance(x, float) and math.isfinite(x) else x
    for f in frames:     # ring bars to 3 decimals and hud readouts to 2 (they are drawn, not read); whole numbers as ints
        rg = f["ring"]
        rg["activity"] = [q(x, 3) for x in rg["activity"]]
        rg["drives"] = {k: [q(x, 3) for x in v] for k, v in rg["drives"].items()}
        f["hud"] = {k: q(v, 2) for k, v in f["hud"].items()}
    st = stamp(param_source=src)
    write_run(name, label, "ring", V2_SEED, meta_sc, world, frames, out, src, st,
              "Illustration: recorded by scripts/export_runs.py with the Python reference on the first v2 test seed by a fixed rule; not a result.")
    print(f"{name}: {len(frames)} frames, t_end {out['time_s']:.1f} s, full success {out.get('success_full')}, "
          f"{(OUT / f'{name}.json').stat().st_size / 1e6:.2f} MB, {time.time() - t0:.0f} s")


def bundle():
    """manifest.json + bundle.js from the recorded files: Study 2 replays first, then the kept Study 1 pair."""
    names = [(e[0], e[-1]) for e in SHOW_V2] + [(e[0], e[-1]) for e in SHOW]
    man, runs = [], {}
    for name, label in names:
        f = f"{name}.json"
        if not (OUT / f).exists():
            print(f"skip {f}: not recorded yet"); continue
        run = json.loads((OUT / f).read_text())
        if run["meta"].get("label") != label:              # labels live in this script; keep the file in step
            run["meta"]["label"] = label
            (OUT / f).write_text(json.dumps(run, separators=(",", ":")))
        runs[f] = run
        man.append({"file": f, "label": f"{label} · seed {run['meta']['seed']}", "study": 2 if name.startswith("v2_") else 1})
    st = stamp(param_source="see each run's meta.params_source")
    manifest = {"runs": man, "_source": "scripts/export_runs.py", **{k: st[k] for k in ("git", "sim_version", "created")},
                "note": "Illustrations, not results."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "bundle.js").write_text("window.RUN_BUNDLE = " + json.dumps({"manifest": manifest, "runs": runs}, separators=(",", ":")) + ";\n")
    return manifest


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


def selfcheck(manifest):
    # files round-trip, every run has frames in time order with the C6 fields the app reads, the eye decodes to
    # w*h*3 bytes, the last frame carries the outcome, v2 runs carry the hud/home fields, and the total stays modest
    total = sum((OUT / r["file"]).stat().st_size for r in manifest["runs"]) + (OUT / "bundle.js").stat().st_size
    assert len(manifest["runs"]) == len(SHOW) + len(SHOW_V2), [r["file"] for r in manifest["runs"]]
    assert total < 15e6, total
    for r in manifest["runs"]:
        run = json.loads((OUT / r["file"]).read_text())
        fr = run["frames"]
        assert run["meta"]["seed"] >= 1000 and run["world"]["objects"] and len(fr) > 10
        if r["file"].startswith("v2_"):
            assert run["meta"]["seed"] == V2_SEED and run["meta"]["scenario"]["brain"]["version"] == 2
            assert all("hud" in f and "home" in f for f in fr) and "home_pad" in run["world"]
            assert run["meta"]["params_source"].startswith("results/tuned_v2_params.json")
        assert all(b["t"] > a["t"] for a, b in zip(fr, fr[1:]))
        assert all({"quad", "pi", "ring", "arbiter", "objects", "mind", "phase"} <= set(f) for f in fr)
        assert all(math.isfinite(f["pi"]["heading_est_deg"]) for f in fr)
        e = next(f["eye"] for f in fr if "eye" in f)
        assert len(base64.b64decode(e["b64"])) == e["w"] * e["h"] * 3
        assert fr[-1]["outcome"] == run["meta"]["outcome"]
    o = {r["file"]: json.loads((OUT / r["file"]).read_text())["meta"]["outcome"] for r in manifest["runs"]}
    assert o["decoy_ring_1010.json"]["wrong_object_stop"] != o["decoy_sm_1010.json"]["wrong_object_stop"], "decoy pair no longer contrasts"
    print(f"export_runs self-check OK; {len(manifest['runs'])} runs, {total / 1e6:.1f} MB on disk (runs + bundle)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bundle"
    if cmd == "record":
        record_v2(sys.argv[2])
    elif cmd == "v1":
        main()
        selfcheck(bundle())
    else:
        selfcheck(bundle())
