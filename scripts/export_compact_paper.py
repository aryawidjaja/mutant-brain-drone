"""Compact paper: every number and figure of paper/compact/main.tex, read from results/ (no simulation).

Writes paper/compact/numbers.tex (one \\newcommand per number used in the text), paper/compact/numbers_sources.json
(macro -> "results/<file>:<key path>" or "derived: <formula>") and paper/compact/figures/fig*.{pdf,png}.
Self-check (__main__): clears every cache, re-reads every source and asserts each macro reproduces; recomputes key
intervals from the run-record parquets (Wilson for proportions; paired bootstrap 10,000, seed 0, scripts/stats.py)
and asserts they equal the stored summaries.
Usage: uv run python scripts/export_compact_paper.py
"""
import json, re, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import arm_permutation_test, boot_idx, paired_bootstrap, paired_diff  # noqa: E402
from run_3d import FAULT_SEV, auc_table  # noqa: E402
from run_v2 import MIDDLE, full, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/compact"
FIG = OUT / "figures"
FAULTS = list(FAULT_SEV)
FAULT_LABEL = {"none": "no fault", "occlusion": "occlusion", "decoy": "look-alike", "compass_half": "half compass",
               "eye_columns": "eye columns", "false_sun": "false sun", "speed_scale": "speed scale", "gust": "gust"}

# ---------------------------------------------------------------- loaders (cached; the self-check clears them)

@lru_cache(None)
def J(f):
    return json.loads((ROOT / "results" / f).read_text())


def jpath(f, path):
    x = J(f)
    for k in path.split("/"):
        x = x[int(k)] if isinstance(x, list) else x[k]
    return x


@lru_cache(None)
def runs(name):
    return pd.read_parquet(ROOT / "results/runs" / f"{name}.parquet")


@lru_cache(None)
def s1():
    """Study 1 3D test records with the severity fraction restored from the level index (auc_table's input)."""
    df = runs("mission_test").copy()
    df["severity"] = [np.nan if f == "none" else FAULT_SEV[f][lv - 1] for f, lv in zip(df.fault, df.level)]
    return df


@lru_cache(None)
def s1_auc(arm):
    return auc_table(s1()[s1().arbiter == arm]).sort_index()


@lru_cache(None)
def s2():
    df = runs("mission_v2_test").copy()
    df.loc[df.fault == "none", "severity"] = np.nan
    df["label"] = df.unit.str.split("__").str[0]
    return df


def s2_arm(label):
    return s2()[s2().label == label]


@lru_cache(None)
def s2_auc(label, own=False):
    """Per-seed, per-fault AUC; full success (target + landed on pad + no collision), or v1's own 2 m definition."""
    g = s2_arm(label)
    return auc_table(g if own else full(g)).sort_index()


def s2_agg(label, own=False):
    return s2_auc(label, own).mean(axis=1).to_numpy()


def s2_cond(label, fault, sev, own=False):
    g = s2_arm(label)
    g = g[(g.fault == fault) & (np.isnan(g.severity) if sev is None else np.isclose(g.severity, sev))]
    ok = (g.success_target & g.success_home & (g.collisions == 0)) if own else g.success_full
    return int(ok.sum()), len(g)


# ---------------------------------------------------------------- number registry

NUM = {}


def num(v, d):
    s = f"{v:.{d}f}"
    if s.startswith("-"):
        s = "\\ensuremath{-}" + s[1:]
    return s


def pfmt(p):
    return "\\ensuremath{<}\\,0.001" if p < 0.001 else f"\\ensuremath{{=}}\\,{p:.{3 if p < 0.05 else 2}f}"


def put(name, fn, d, source, fmt=None):
    assert re.fullmatch(r"[A-Za-z]+", name) and name not in NUM, name
    v = fn()
    NUM[name] = {"fn": fn, "d": d, "fmt": fmt, "source": source, "value": v,
                 "text": fmt(v) if fmt else (f"{int(v):,}".replace(",", "{,}") if d == "int" else num(v, d))}


def R(name, f, path, d=3, fmt=None):
    put(name, lambda: jpath(f, path), d, f"results/{f}:{path}", fmt)


def Dv(name, fn, d, formula, fmt=None):
    put(name, fn, d, f"derived: {formula}", fmt)


def ci(name, f, base, d=3, keys=("est", "lo", "hi")):
    for suf, k in zip(("", "Lo", "Hi"), keys):
        R(name + suf, f, f"{base}/{k}", d)


def register():
    NUM.clear()
    # ---- Study 1: 2D kill tests
    R("SoneDstar", "kt1.json", "primary/D_star_deg", 1)
    R("SoneDstarFine", "kt1.json", "posthoc_fine_grid/bracket_D_star_deg/0", 1)
    R("SoneDstarExclLo", "kt1.json", "posthoc_fine_grid/bracket_D_star_excl_ratio1_deg/0", 1)
    R("SoneDstarExclHi", "kt1.json", "posthoc_fine_grid/bracket_D_star_excl_ratio1_deg/1", 1)
    ci("SoneKTtwoDiff", "kt2.json", "ring_minus_best_other")
    R("SoneKTtwoPermP", "kt2.json", "posthoc_permutation/p_two_sided", fmt=pfmt)
    R("SoneKTtwoGain", "kt2.json", "tuned_params/ring/ring_gain", 3)
    rp = lambda: J("kt3.json")["points"]["ring"]  # noqa: E731
    R("SoneHoldGzero", "kt3.json", "points/ring/0/hold_mean_s", 2)
    R("SoneRelGzero", "kt3.json", "points/ring/0/release_mean_s", 2)
    Dv("SoneHoldMax", lambda: max(p["hold_mean_s"] for p in rp()), 2, "max over ring_gain of results/kt3.json:points/ring/*/hold_mean_s")
    Dv("SoneCapGain", lambda: min(p["value"] for p in rp() if p["release_mean_s"] >= 20.0), 1,
       "smallest ring_gain in results/kt3.json:points/ring with release_mean_s at the 20 s cap")
    for suf, k in (("", "delta_s"), ("Lo", "lo"), ("Hi", "hi")):
        R("SoneDaccGone" + suf, "kt3.json", f"ring_vs_rival/accumulator/1/{k}", 2)
    Dv("SoneDaccMin", lambda: min(x["delta_s"] for x in J("kt3.json")["ring_vs_rival"]["accumulator"] if x["delta_s"] is not None), 1,
       "min over gains of results/kt3.json:ring_vs_rival/accumulator/*/delta_s")
    R("SoneSMUnclamped", "kt3.json", "posthoc_unclamped/points/state_machine/0/release_signed_mean_s", 2)
    R("SoneSMRatio", "kt3.json", "posthoc_unclamped/points/state_machine/0/ratio_at_switch_mean", 2)
    eb = "best_ring_vs_rival/random64/accumulator|pareto_interp"
    R("SoneEnvHold", "kt3_envelope.json", f"{eb}/best_ring_hold_s", 2)
    R("SoneEnvRel", "kt3_envelope.json", f"{eb}/best_ring_release_s", 2)
    ci("SoneEnvDacc", "kt3_envelope.json", eb, 2, ("delta_s", "lo", "hi"))
    R("SoneEnvDaccPooled", "kt3_envelope.json", "best_ring_vs_rival/random64_plus_kt3_sweep/accumulator|pareto_interp/delta_s", 2)
    R("SoneEnvHighGain", "kt3_envelope.json", "ring_high_gain/n_configs_gain_ge_0.7", "int")
    # ---- Study 1: 3D mission
    for a, n in (("ring", "Ring"), ("state_machine", "SM"), ("accumulator", "Acc"), ("vote", "Vote"), ("sum", "Sum"), ("ring_static", "Static")):
        R("SoneAUC" + n, "mission_h1.json", f"aggregate_auc/{a}/mean", 3)
    R("SoneAUCRingLo", "mission_h1.json", "aggregate_auc/ring/lo", 3)
    R("SoneAUCRingHi", "mission_h1.json", "aggregate_auc/ring/hi", 3)
    ci("SoneHoneDiff", "mission_h1.json", "ring_minus_best_other", 4)
    R("SoneHonePermP", "mission_h1.json", "permutation_test/p_two_sided", fmt=pfmt)
    R("SoneHoneSumP", "mission_diag.json", "h1/each_arm_minus_best_other_permutation/sum/p_two_sided", fmt=pfmt)
    R("SoneHoneStaticP", "mission_diag.json", "h1/each_arm_minus_best_other_permutation/ring_static/p_two_sided", fmt=pfmt)
    nd = lambda: J("mission_diag.json")["h1"]["aggregate_auc_without_decoy_fault"].values()  # noqa: E731
    Dv("SoneNoDecoyLo", lambda: min(nd()), 3, "min over arms of results/mission_diag.json:h1/aggregate_auc_without_decoy_fault")
    Dv("SoneNoDecoyHi", lambda: max(nd()), 3, "max over arms of results/mission_diag.json:h1/aggregate_auc_without_decoy_fault")
    R("SoneDecoyRing", "mission_h1.json", "per_fault_auc/ring/decoy", 3)
    R("SoneDecoySM", "mission_h1.json", "per_fault_auc/state_machine/decoy", 3)
    R("SoneFalseSun", "mission_h1.json", "per_fault_auc/ring/false_sun", 2)
    R("SoneSpeed", "mission_h1.json", "per_fault_auc/ring/speed_scale", 2)
    R("SoneCompass", "mission_h1.json", "per_fault_auc/ring/compass_half", 2)
    R("SoneThreeDGain", "mission_h1.json", "tuned_params/ring/ring_gain", 3)
    R("SoneNoFaultRing", "mission_h1.json", "success_by_condition/ring/none:None", 2)
    Dv("SoneNmissions", lambda: len(runs("mission_test")), "int", "rows of results/runs/mission_test.parquet")
    hp = lambda: J("mission_h3.json")["points"]  # noqa: E731
    Dv("SoneHthreePeak", lambda: max(p["hold_mean_s"] for p in hp()["ring"]), 2, "max of results/mission_h3.json:points/ring/*/hold_mean_s")
    Dv("SoneHthreePeakGain", lambda: max(hp()["ring"], key=lambda p: p["hold_mean_s"])["value"], 1,
       "ring_gain at the max of results/mission_h3.json:points/ring/*/hold_mean_s")
    Dv("SoneHthreeCeil", lambda: max(p["hold_mean_s"] for a in hp().values() for p in a), 2,
       "max over every arm and sweep point of results/mission_h3.json:points/*/*/hold_mean_s")
    Dv("SoneUsableTwoD", lambda: sum(x > 0.5 for x in J("kt2_tuned_params.json")["arms"]["ring"]["rung1_auc"]), "int",
       "count of results/kt2_tuned_params.json:arms/ring/rung1_auc > 0.5")
    Dv("SoneUsableThreeD", lambda: sum(x > 0 for x in J("tune3d_r1.json")["scores"]["ring"].values()), "int",
       "count of results/tune3d_r1.json:scores/ring/* > 0")
    for suf, k in (("", "delta_s"), ("Lo", "lo"), ("Hi", "hi")):
        R("SoneHthreeAccGzero" + suf, "mission_h3.json", f"ring_vs_rival/accumulator/0/{k}", 2)
    R("SoneHthreeStopGone", "mission_diag.json", "h3/ring/10/occlusion_target_stop_rate", 2)
    R("SoneHthreeLimitGone", "mission_diag.json", "h3/ring/10/occlusion_time_limit_rate", 2)
    # ---- Study 1: recognition (H4)
    for suf, k in (("", "auroc"), ("Lo", "ci95_lo"), ("Hi", "ci95_hi")):
        R("SoneMB" + suf, "recog_auroc.json", f"recognisers/mb/lf_off/{k}/0", 3)
    R("SonePix", "recog_auroc.json", "recognisers/pixels/lf_off/auroc/0", 3)
    R("SoneDinoOff", "recog_auroc.json", "recognisers/dino_patch/lf_off/auroc/0", 3)
    R("SoneDinoOn", "recog_auroc.json", "recognisers/dino_patch/lf_on/auroc/0", 3)
    # ---- Study 2: components
    ci("StwoHeadRing", "kt5.json", "aggregate_fault_set/rms_deg/ring", 2)
    R("StwoHeadKF", "kt5.json", "aggregate_fault_set/rms_deg/kf/est", 2)
    R("StwoHeadCF", "kt5.json", "aggregate_fault_set/rms_deg/cf/est", 2)
    R("StwoHeadVone", "kt5.json", "aggregate_fault_set/rms_deg/v1/est", 1)
    ci("StwoHeadKFdiff", "kt5.json", "aggregate_fault_set/delta_vs_ring_deg/kf", 2)
    for e, n in (("ring", "Ring"), ("cf", "CF"), ("kf", "KF")):
        R("StwoCompass" + n, "kt5.json", f"per_condition/compass_half/rms_deg/{e}/est", 2)
    R("StwoKTsevenGated", "kt7.json", "kt2_world/aggregate_auc/ring_gated/mean", 3)
    R("StwoKTsevenVote", "kt7.json", "kt2_world/aggregate_auc/vote_vg/mean", 3)
    R("StwoKTsevenP", "kt7.json", "kt2_world/ring_gated_minus_best_other_permutation/p_two_sided", fmt=pfmt)
    R("StwoSMtimeOff", "kt7.json", "premature_switch_noninferiority/rival_tuned_time_off_s/state_machine", 3)
    to = lambda: [x["est"] for x in J("kt7.json")["premature_switch_noninferiority"]["ring_minus_rival"]["state_machine"] if x["ring_gain"] <= 0.7]  # noqa: E731
    Dv("StwoTimeOffLo", lambda: min(to()), 1, "min over ring_gain <= 0.7 of results/kt7.json:premature_switch_noninferiority/ring_minus_rival/state_machine/*/est")
    Dv("StwoTimeOffHi", lambda: max(to()), 1, "max over ring_gain <= 0.7 of the same")
    R("StwoFront", "recog_v2_gate.json", "auroc_0deg", 3)
    R("StwoFrontLo", "recog_v2_gate.json", "auroc_0deg_ci95/0", 3)
    R("StwoFrontHi", "recog_v2_gate.json", "auroc_0deg_ci95/1", 3)
    R("StwoMargin", "recog_v2_gate.json", "margin_over_pixels_0deg", 3)
    R("StwoMarginLo", "recog_v2_gate.json", "margin_ci95/0", 3)
    R("StwoMarginHi", "recog_v2_gate.json", "margin_ci95/1", 3)
    R("StwoNN", "recog_v2_auroc.json", "arms/nn_front/ms/auroc/0", 3)
    R("StwoDino", "recog_v2_auroc.json", "arms/dino_patch/ms/auroc/0", 3)
    R("StwoMBvone", "recog_v2_auroc.json", "arms/mb_v1/lf13/auroc/0", 3)
    R("StwoFrontNinety", "recog_v2_auroc.json", "arms/mb_front/ms/auroc/6", 2)
    for suf, k in (("", "by_theta"), ("Lo", "ci95_lo"), ("Hi", "ci95_hi")):
        R("StwoMSvsGoal" + suf, "recog_v2_auroc.json", f"paired_differences/mb_front_ms_minus_mb_front_goal/{k}/0", 3)
    # ---- Study 2: integrated mission (H9)
    R("Nseeds", "mission_v2.json", "n_seeds", "int")
    Dv("StwoNmissions", lambda: len(runs("mission_v2_test")), "int", "rows of results/runs/mission_v2_test.parquet")
    Dv("StwoNarms", lambda: len(J("mission_v2.json")["arms"]), "int", "len(results/mission_v2.json:arms)")
    R("StwoNoFault", "mission_v2.json", "H9_i/success", 3)
    R("StwoNoFaultLo", "mission_v2.json", "H9_i/wilson95/0", 3)
    R("StwoNoFaultHi", "mission_v2.json", "H9_i/wilson95/1", 3)
    for suf, k in (("", "mean"), ("Lo", "lo"), ("Hi", "hi")):
        R("StwoAUC" + suf, "mission_v2.json", f"aggregate_auc/v2_ring/{k}", 3)
    R("StwoAUCvone", "mission_v2.json", "aggregate_auc/v1/mean", 3)
    ci("StwoDiffVone", "mission_v2.json", "H9_ii/diff")
    R("StwoVoneOwn", "mission_v2.json", "H9_ii/secondary_v1_own_definition/auc_v1_own", 3)
    ci("StwoDiffOwn", "mission_v2.json", "H9_ii/secondary_v1_own_definition/diff")
    mids = {"occlusion": "Occ", "decoy": "Decoy", "compass_half": "Compass", "eye_columns": "Eye", "false_sun": "Sun",
            "speed_scale": "Speed", "gust": "Gust"}
    for f, n in mids.items():
        R("StwoMid" + n, "mission_v2.json", f"H9_iii/per_fault/{f}/success", 2)
    Dv("StwoMidMin", lambda: min(m["success"] for f, m in J("mission_v2.json")["H9_iii"]["per_fault"].items() if f != "decoy"), 2,
       "min over faults other than decoy of results/mission_v2.json:H9_iii/per_fault/*/success")
    dec = lambda: [v for k, v in J("mission_v2.json")["success_by_condition"]["v2_ring"].items() if k.startswith("decoy:")]  # noqa: E731
    Dv("StwoDecoyLo", lambda: min(dec()), 2, "min over decoy levels of results/mission_v2.json:success_by_condition/v2_ring/decoy:*")
    Dv("StwoDecoyHi", lambda: max(dec()), 2, "max over decoy levels of the same")
    abl = {"landing": "Landing", "home": "Home", "heading": "Heading", "front": "Front", "gated_arbiter": "Gated", "inspect": "Inspect"}
    for m, n in abl.items():
        ci("StwoAbl" + n, "mission_v2.json", f"ablations_v2_ring_minus_arm/v2_ring_minus_{m}")
    R("StwoMinusHomeAUC", "mission_v2.json", "aggregate_auc/v2_ring_minus_home/mean", 3)
    R("StwoMinusHeadingAUC", "mission_v2.json", "aggregate_auc/v2_ring_minus_heading/mean", 3)
    R("StwoKF", "mission_v2.json", "aggregate_auc/v2_ring_kf/mean", 3)
    kfd = lambda k: (lambda: paired_diff(s2_agg("v2_ring"), s2_agg("v2_ring_kf"))[k])  # noqa: E731
    for suf, k in (("", "est"), ("Lo", "lo"), ("Hi", "hi")):
        Dv("StwoRingMinusKF" + suf, kfd(k), 3, f"stats.paired_diff(per-seed AUC v2_ring, v2_ring_kf)[{k}] from results/runs/mission_v2_test.parquet (10,000, seed 0)")
    R("StwoPermP", "mission_v2.json", "v2_ring_vs_rivals/permutation_test/p_two_sided", fmt=pfmt)
    R("StwoVsVote", "mission_v2.json", "v2_ring_vs_rivals/pairwise/v2_vote/est", 3)
    R("StwoVsVoteP", "mission_v2.json", "v2_ring_vs_rivals/pairwise/v2_vote/p_holm", fmt=pfmt)
    R("StwoAUCsm", "mission_v2.json", "aggregate_auc/v2_state_machine/mean", 3)
    riv = lambda: [J("mission_v2.json")["aggregate_auc"][f"v2_{a}"]["mean"] for a in ("ring_static", "vote", "sum", "accumulator", "state_machine")]  # noqa: E731
    Dv("StwoRivalLo", lambda: min(riv()), 3, "min of results/mission_v2.json:aggregate_auc/v2_{ring_static,vote,sum,accumulator,state_machine}/mean")
    Dv("StwoRivalHi", lambda: max(riv()), 3, "max of the same")
    R("StwoTunedGain", "tuned_v2_params.json", "arms/ring/params/ring_gain", 2)
    R("StwoLFzero", "mission_v2.json", "aggregate_auc/v2_ring_lf0/mean", 3)
    Dv("StwoLFzeroStops", lambda: int(s2_arm("v2_ring_lf0").query("fault == 'none'").success_target.sum()), "int",
       "target stops in no-fault v2_ring_lf0 rows of results/runs/mission_v2_test.parquet")
    R("StwoLFzeroColl", "mission_v2.json", "collision_rate/v2_ring_lf0", 2)
    R("StwoNoTip", "mission_v2.json", "aggregate_auc/v2_ring_no_tip/mean", 3)
    R("StwoNoTipNoFault", "mission_v2.json", "success_by_condition/v2_ring_no_tip/none:None", 2)
    R("StwoPoorTip", "mission_v2.json", "aggregate_auc/v2_ring_poor_tip/mean", 3)
    R("StwoPoorTipNoFault", "mission_v2.json", "success_by_condition/v2_ring_poor_tip/none:None", 2)
    R("StwoLandErr", "mission_v2.json", "landing/v2_ring/none/landing_error_median_m", 3)
    R("StwoTouch", "mission_v2.json", "landing/v2_ring/none/touchdown_speed_median_mps", 2)
    R("StwoGustPad", "mission_v2.json", "landing/v2_ring/gust/landed_on_pad", 2)
    R("StwoGustErr", "mission_v2.json", "landing/v2_ring/gust/landing_error_median_m", 2)
    R("StwoPIPad", "mission_v2.json", "landing/v2_ring_minus_home/none/landed_on_pad", 2)
    R("StwoPIErr", "mission_v2.json", "landing/v2_ring_minus_home/none/landing_error_median_m", 2)
    R("StwoWrongRing", "mission_v2.json", "wrong_object_stop_rate/v2_ring", 3)
    R("StwoWrongSM", "mission_v2.json", "wrong_object_stop_rate/v2_state_machine", 3)
    R("StwoWrongAcc", "mission_v2.json", "wrong_object_stop_rate/v2_accumulator", 3)
    R("StwoCollRing", "mission_v2.json", "collision_rate/v2_ring", 3)
    R("StwoAltSearch", "v2_alt_decision.json", "ALT_SEARCH", 1)


def write_numbers():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/export_compact_paper.py from results/; do not edit. Sources: numbers_sources.json"]
    lines += [f"\\newcommand{{\\{k}}}{{{v['text']}}}" for k, v in NUM.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    src = {k: {"source": v["source"], "value": v["value"], "text": v["text"]} for k, v in NUM.items()}
    (OUT / "numbers_sources.json").write_text(json.dumps(src, indent=1, default=float))


# ---------------------------------------------------------------- figures

ORANGE = "#E66100"
BLUES = ["#08519C", "#3182BD", "#6BAED6", "#9ECAE1", "#C6DBEF"]
GREYS = ["#4D4D4D", "#8C8C8C", "#BDBDBD"]
S1_ARMS = [("ring", "Ring hub", ORANGE), ("ring_static", "Ring, recurrence off", GREYS[2]), ("vote", "Static vote", GREYS[1]),
           ("sum", "Vector sum", BLUES[2]), ("accumulator", "Leaky accumulator", BLUES[1]), ("state_machine", "State machine", BLUES[0])]
W1, W2 = 3.4, 7.0


def style():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [f for f in ("Arial", "DejaVu Sans") if f in have],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def boot_mean_ci(x, n_boot=10_000):
    x = np.asarray(x, float)
    bs = x[boot_idx(len(x), n_boot, 0)].mean(axis=1)
    return np.quantile(bs, [0.025, 0.975])


def fig1():
    """System diagram: which part comes from which model, and what Study 2 added."""
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 53); ax.axis("off")
    INS, ENG = "#FBE3CF", "#ECECEC"

    def box(x, y, w, h, title, sub, fc, dashed=False, ec="#333333", lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                    ls=(0, (3, 2)) if dashed else "-"))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=6.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.0, sub, ha="center", va="top", fontsize=5.6, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=0, shrinkB=0, mutation_scale=7))

    box(0.5, 37, 14, 14, "Eye", "ray-cast panorama,\nlow resolution", ENG)
    box(0.5, 20, 14, 14, "Compass", "sun compass;\ngyro yaw rate\n(Study 2)", ENG)
    box(0.5, 3, 14, 14, "Odometry", "body speed", ENG)
    box(19, 37, 23, 14, "Mushroom body", "one-shot target familiarity\n(after Ardin et al.); Study 2:\ncolour-opponent and edge\nfront end", INS)
    box(19, 20, 23, 14, "Heading ring", "EPG-style bump moved by the\nyaw rate, corrected by the\nsun through a conflict gate\n(Study 2)", INS, dashed=True)
    box(19, 3, 23, 14, "Path integrator", "central-complex home vector\n(after Stone et al.,\nreimplemented)", INS)
    box(47, 3, 22, 14, "Home view and landing", "nest learning flight, view-\nbased final approach,\ndescent onto the pad\n(Study 2)", INS, dashed=True)
    box(47, 25, 22, 22, "Steering hub", "ring attractor (after Sun et al.)\nor one of five rival arbiters\nwith the same inputs and the\nsame tuning budget;\nStudy 2: evidence-gated ring,\nvisibility channel for all", INS, ec=ORANGE, lw=1.6)
    box(74, 25, 12, 22, "Avoidance", "optic-flow\nbalance and\nlooming, added\nafter the hub\n(descending\nlevel)", ENG)
    box(89.5, 25, 10, 22, "Flight\ncontrol", "\ncascade PID,\nmotors,\n6-DOF\nCrazyflie\nmodel", ENG)
    arrow(14.9, 44, 18.6, 44); arrow(14.9, 27, 18.6, 27); arrow(14.9, 10, 18.6, 10)
    arrow(14.9, 24, 18.6, 13)
    arrow(42.4, 44, 46.6, 40)
    arrow(42.4, 27, 46.6, 31)
    arrow(42.4, 10, 46.6, 10)
    arrow(58, 17.4, 58, 24.6)
    arrow(69.4, 36, 73.6, 36)
    arrow(86.4, 36, 89.1, 36)
    leg = [FancyBboxPatch((0, 0), 1, 1, fc=INS, ec="#333333", lw=0.8), FancyBboxPatch((0, 0), 1, 1, fc=ENG, ec="#333333", lw=0.8),
           FancyBboxPatch((0, 0), 1, 1, fc="white", ec="#333333", lw=0.8, ls=(0, (3, 2))), FancyBboxPatch((0, 0), 1, 1, fc="white", ec=ORANGE, lw=1.6)]
    ax.legend(leg, ["insect model (reimplemented)", "engineered part", "added in Study 2", "arbiter under test"],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1, fontsize=6.0, handlelength=1.6)
    save(fig, "fig1_system")


def fig2():
    """Study 1 KT3 stubbornness frontier (test seeds) and the post hoc 64-configuration envelope (training seeds)."""
    k3, env = J("kt3.json")["points"], J("kt3_envelope.json")["configs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.6))
    for arm, lab, c, mk in (("ring", "Ring hub (gain swept)", ORANGE, "o"), ("accumulator", "Leaky accumulator (leak swept)", BLUES[1], "s"),
                            ("state_machine", "State machine (timeout swept)", BLUES[0], "D")):
        p = k3[arm]
        h = np.array([q["hold_mean_s"] for q in p]); r = np.array([q["release_mean_s"] for q in p])
        a.errorbar(h, r, xerr=1.96 * np.array([q["hold_sem"] for q in p]), yerr=1.96 * np.array([q["release_sem"] for q in p]),
                   color=c, marker=mk, ms=3, lw=1.0, elinewidth=0.6, capsize=0, label=lab)
    for arm, lab, c, mk in (("vote", "Static vote", GREYS[1], "^"), ("sum", "Vector sum", BLUES[2], "v"), ("ring_static", "Ring, recurrence off", GREYS[2], "P")):
        q = k3[arm][0]
        a.errorbar(q["hold_mean_s"], q["release_mean_s"], xerr=1.96 * q["hold_sem"], yerr=1.96 * q["release_sem"], color=c,
                   marker=mk, ms=4.5, ls="none", elinewidth=0.6, label=lab, mec="#333333", mew=0.4)
    for q in k3["ring"]:
        if q["value"] in (0.0, 0.5, 0.7, 3.0):
            a.annotate(f"g = {q['value']:g}", (q["hold_mean_s"], q["release_mean_s"]), xytext=(4, -9 if q["value"] == 3.0 else 3),
                       textcoords="offset points", fontsize=6, color=ORANGE)
    a.axhline(20, color="#999999", lw=0.5, ls=":")
    a.text(0.2, 20.4, "release cap (20 s)", fontsize=6, color="#777777")
    a.set_xlabel("hold of a hidden target (s)"); a.set_ylabel("release of a fading look-alike (s)")
    a.set_xlim(-0.3, 10.6); a.set_ylim(-0.8, 22)
    a.set_title("Pre-registered sweep, test seeds", loc="left")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6)
    panel(a, "a")
    for arm, lab, c, mk in (("ring", "Ring hub", ORANGE, "o"), ("accumulator", "Leaky accumulator", BLUES[1], "s"), ("state_machine", "State machine", BLUES[0], "D")):
        pts = np.array([(q["hold_mean_s"], q["release_mean_s"]) for q in env[arm] if q["source"] == "random_config"])
        b.scatter(pts[:, 0], pts[:, 1], s=7, color=c, alpha=0.55, lw=0, marker=mk, label=f"{lab}, 64 configurations")
        o = pts[np.lexsort((pts[:, 1], -pts[:, 0]))]          # hold descending; Pareto front = running min of release
        front, best = [], np.inf
        for h, r in o:
            if r < best:
                front.append((h, r)); best = r
        f = np.array(front)
        b.step(f[:, 0], f[:, 1], where="post", color=c, lw=1.0)
    bb = J("kt3_envelope.json")["best_ring_vs_rival"]["random64"]["accumulator|pareto_interp"]
    b.scatter([bb["best_ring_hold_s"]], [bb["best_ring_release_s"]], s=40, facecolor="none", edgecolor=ORANGE, lw=1.2, zorder=5)
    b.annotate("best ring configuration", (bb["best_ring_hold_s"], bb["best_ring_release_s"]), xytext=(-82, 18),
               textcoords="offset points", fontsize=6, color=ORANGE, arrowprops=dict(arrowstyle="-", lw=0.5, color=ORANGE))
    b.set_xlabel("hold of a hidden target (s)"); b.set_ylabel("release of a fading look-alike (s)")
    b.set_xlim(-0.3, 10.6); b.set_ylim(-0.8, 22)
    b.set_title("Post hoc envelope, training seeds", loc="left")
    b.legend(loc="upper left", fontsize=6)
    panel(b, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_kt3")


def fig3():
    """Study 1 3D mission: per-fault AUC per arm with bootstrap CIs over seeds."""
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    w = 0.13
    for i, (arm, lab, c) in enumerate(S1_ARMS):
        t = s1_auc(arm)
        m = t.mean().reindex(FAULTS).to_numpy()
        lo, hi = np.array([boot_mean_ci(t[f]) for f in FAULTS]).T
        x = np.arange(len(FAULTS)) + (i - 2.5) * w
        ax.bar(x, m, w, color=c, edgecolor="#333333", lw=0.3, label=lab)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#222222", elinewidth=0.6)
    ax.set_xticks(np.arange(len(FAULTS)), [FAULT_LABEL[f] for f in FAULTS])
    ax.set_ylabel("success-vs-severity AUC"); ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=6.4)
    save(fig, "fig3_s1_mission")


def fig4():
    """Recognition AUROC vs azimuth on the Study 2 test layouts, with layout-bootstrap CIs."""
    d = J("recog_v2_auroc.json")
    th = np.array(d["thetas"])
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    ax.axvspan(0, 45, color="#F2F2F2", lw=0)
    ax.text(22.5, 0.255, "flown arc", ha="center", fontsize=6, color="#777777")
    for arm, ms, lab, c, ls in (("mb_v1", "lf13", "Study 1 mushroom body, grey pixels", GREYS[1], "--"),
                                ("pixels", "ms", "raw pixels", GREYS[2], "-"),
                                ("nn_front", "ms", "front-end features, nearest view", BLUES[2], "-"),
                                ("dino_patch", "ms", "DINOv2 patches (object-centred memory)", BLUES[0], "-"),
                                ("mb_front", "ms", "Study 2 mushroom body (front end)", ORANGE, "-")):
        x = d["arms"][arm][ms]
        ax.fill_between(th, x["ci95_lo"], x["ci95_hi"], color=c, alpha=0.22, lw=0)
        ax.plot(th, x["auroc"], color=c, ls=ls, marker="o", ms=2.2, label=lab)
    ax.axhline(0.75, color="#333333", lw=0.6, ls="--"); ax.axhline(0.5, color="#999999", lw=0.5, ls=":")
    ax.text(100, 0.755, "0.75 criterion", fontsize=5.8, ha="center", va="bottom")
    ax.set_xlabel("probe azimuth from the stored view (deg)"); ax.set_ylabel("AUROC, target vs look-alikes")
    ax.set_xticks(range(0, 181, 30)); ax.set_xlim(-4, 184); ax.set_ylim(0.24, 0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), fontsize=5.8, ncol=1)
    save(fig, "fig4_recognition")


S2_ARMS = [("v2_ring", "Study 2 brain (gated ring)", ORANGE, None), ("v2_ring_static", "ring, recurrence off", GREYS[2], None),
           ("v2_vote", "static vote", GREYS[1], None), ("v2_sum", "vector sum", BLUES[2], None),
           ("v2_accumulator", "leaky accumulator", BLUES[1], None), ("v2_state_machine", "state machine", BLUES[0], None),
           ("v1", "Study 1 brain (own 2 m success)", GREYS[0], "////")]


def fig5():
    """Study 2 headline: success at no fault and at each fault's middle severity, Wilson CIs, criterion lines."""
    conds = [("none", None)] + [(f, MIDDLE[f]) for f in FAULTS]
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    w = 0.115
    for i, (arm, lab, c, hatch) in enumerate(S2_ARMS):
        own = arm == "v1"
        ks = [s2_cond(arm, f, v, own) for f, v in conds]
        p = np.array([k / n for k, n in ks]); ci_ = np.array([wilson(k, n) for k, n in ks])
        x = np.arange(len(conds)) + (i - 3) * w
        ax.bar(x, p, w, color=c if not hatch else "white", edgecolor=c if hatch else "#333333", hatch=hatch, lw=0.4 if not hatch else 0.6, label=lab)
        ax.errorbar(x, p, yerr=[p - ci_[:, 0], ci_[:, 1] - p], fmt="none", ecolor="#222222", elinewidth=0.6)
    ax.plot([-0.5, 0.5], [0.9, 0.9], color="#222222", lw=0.8, ls="--")
    ax.plot([0.5, len(conds) - 0.5], [0.5, 0.5], color="#222222", lw=0.8, ls="--")
    ax.plot([], [], color="#222222", lw=0.8, ls="--", label="pre-registered criteria (0.90; 0.50)")
    unit = {"occlusion": "{:g} s", "decoy": "similarity {:g}", "compass_half": "one level", "eye_columns": "{:.0%}",
            "false_sun": "{:g} deg", "speed_scale": "scale error {:g}", "gust": "{}"}
    lv = J("mission_v2.json")["H9_iii"]["per_fault"]
    labels = ["no fault"] + [f"{FAULT_LABEL[f]}\n" + unit[f].format(lv[f]["level"]) for f in FAULTS]
    ax.set_xticks(np.arange(len(conds)), labels)
    ax.set_ylabel("mission success"); ax.set_ylim(0, 1.05); ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=6.3)
    save(fig, "fig5_s2_headline")


ABL = [("v2_ring_minus_landing", "landing controller removed", GREYS[1]), ("v2_ring_minus_home", "home view memory removed", ORANGE),
       ("v2_ring_minus_heading", "heading ring removed (Study 1 compass)", ORANGE), ("v2_ring_kf", "heading ring swapped for Kalman filter", BLUES[1]),
       ("v2_ring_minus_front", "insect front end removed", ORANGE), ("v2_ring_minus_gated_arbiter", "evidence gating removed", ORANGE),
       ("v2_ring_minus_inspect", "inspection rule removed", GREYS[1]),
       ("v2_ring_lf0", "learning flight 0 s (calibrated with flight)", GREYS[2]), ("v2_ring_no_tip", "no operator tip", GREYS[2]),
       ("v2_ring_poor_tip", "poor operator tip", GREYS[2])]


def fig6():
    """Study 2 ablations and sensitivity arms: paired AUC difference (full brain minus arm), 95% paired bootstrap."""
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    base = s2_agg("v2_ring")
    rows = [(lab, c, paired_diff(base, s2_agg(arm))) for arm, lab, c in ABL]
    y = np.arange(len(rows))[::-1].astype(float)
    y[-3:] -= 0.5
    for yy, (lab, c, d) in zip(y, rows):
        ax.barh(yy, d["est"], 0.62, color=c, edgecolor="#333333", lw=0.3)
        ax.errorbar(d["est"], yy, xerr=[[d["est"] - d["lo"]], [d["hi"] - d["est"]]], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=1.5)
        ax.text(max(d["hi"], 0) + 0.015, yy, f"{d['est']:.3f}", va="center", fontsize=6)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.4)
    ax.axvline(0, color="#333333", lw=0.6)
    ax.axhline(y[-3] + 0.75, color="#BBBBBB", lw=0.5, ls=":")
    ax.text(0.98, y[-3] + 0.85, "sensitivity arms", fontsize=5.8, color="#777777", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("AUC lost (full brain minus arm)"); ax.set_xlim(-0.05, 1.0)
    save(fig, "fig6_s2_ablations")


def fig7():
    """Landing: horizontal error at touchdown (ECDF) and touchdown speed, Study 2 brain vs the path-integrator-only home."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1 * 1.55, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    for lab, g, c, ls in (("Study 2 brain, no gust", s2_arm("v2_ring").query("fault != 'gust'"), ORANGE, "-"),
                          ("Study 2 brain, gust", s2_arm("v2_ring").query("fault == 'gust'"), ORANGE, "--"),
                          ("home view memory removed", s2_arm("v2_ring_minus_home"), GREYS[1], "-")):
        e = np.sort(g[g.success_target].landing_error_m.to_numpy())
        a.plot(np.maximum(e, 1e-3), np.arange(1, len(e) + 1) / len(e), color=c, ls=ls, label=f"{lab} (n = {len(e):,})")
    a.axvline(0.5, color="#222222", lw=0.7, ls=":"); a.text(0.4, 0.55, "pad edge 0.5 m", fontsize=5.6, rotation=90, ha="right")
    a.set_xscale("log"); a.set_xlim(1e-3, 30)
    a.set_xlabel("landing error (m)"); a.set_ylabel("cumulative fraction of missions")
    a.legend(loc="upper center", fontsize=5.4, bbox_to_anchor=(0.5, -0.3), ncol=1)
    panel(a, "a")
    bins = np.linspace(0.15, 0.6, 31)
    for lab, g, c in (("no gust", s2_arm("v2_ring").query("fault != 'gust'"), ORANGE), ("gust", s2_arm("v2_ring").query("fault == 'gust'"), BLUES[1])):
        v = g.touchdown_speed_mps.dropna().to_numpy()
        b.hist(v, bins=bins, color=c, alpha=0.75, density=True, label=f"{lab} (n = {len(v):,})", edgecolor="white", lw=0.2)
    b.set_xlabel("touchdown speed (m/s)"); b.set_ylabel("density"); b.legend(loc="upper center", fontsize=5.4, bbox_to_anchor=(0.5, -0.3))
    panel(b, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_landing")


def fig8():
    """Heading: (a) 2D-replay RMS heading error per condition; (b) mission AUC per fault, ring heading vs Kalman vs v1 compass."""
    k5 = J("kt5.json")["per_condition"]
    conds = J("kt5.json")["conditions"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1.15]})
    w = 0.2
    for i, (e, lab, c) in enumerate((("v1", "Study 1 sun compass", GREYS[1]), ("cf", "complementary filter", BLUES[2]),
                                     ("kf", "Kalman filter", BLUES[0]), ("ring", "heading ring", ORANGE))):
        m = np.array([k5[cd]["rms_deg"][e]["est"] for cd in conds])
        lo = np.array([k5[cd]["rms_deg"][e]["lo"] for cd in conds]); hi = np.array([k5[cd]["rms_deg"][e]["hi"] for cd in conds])
        x = np.arange(len(conds)) + (i - 1.5) * w
        a.bar(x, m, w, color=c, edgecolor="#333333", lw=0.3, label=lab)
        a.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#222222", elinewidth=0.6)
    a.set_yscale("log"); a.set_ylim(0.2, 200)
    a.set_xticks(np.arange(len(conds)), [c.replace("false_sun_", "sun\n").replace("compass_half", "half\ncomp.").replace("none", "no\nfault").replace("gust", "gust\n") for c in conds], fontsize=6.3)
    a.set_ylabel("RMS heading error (deg)"); a.legend(ncol=2, loc="upper left", fontsize=5.8)
    a.set_ylim(0.2, 900)
    a.set_title("2D replays of recorded flights", loc="left")
    panel(a, "a")
    cols = FAULTS + ["all"]
    w = 0.26
    for i, (arm, lab, c) in enumerate((("v2_ring_minus_heading", "Study 1 sun compass", GREYS[1]), ("v2_ring_kf", "Kalman filter", BLUES[0]),
                                       ("v2_ring", "heading ring", ORANGE))):
        t = s2_auc(arm).copy(); t["all"] = t[FAULTS].mean(axis=1)
        m = t[cols].mean().to_numpy()
        lo, hi = np.array([boot_mean_ci(t[f]) for f in cols]).T
        x = np.arange(len(cols)) + (i - 1) * w
        b.bar(x, m, w, color=c, edgecolor="#333333", lw=0.3, label=lab)
        b.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#222222", elinewidth=0.6)
    short = {"occlusion": "occlu-\nsion", "decoy": "look-\nalike", "compass_half": "half\ncomp.", "eye_columns": "eye\ncolumns",
             "false_sun": "false\nsun", "speed_scale": "speed\nscale", "gust": "gust\n", "all": "aggre-\ngate"}
    b.set_xticks(np.arange(len(cols)), [short[f] for f in cols], fontsize=6.3)
    b.set_ylabel("full-mission AUC"); b.set_ylim(0, 1.16)
    b.set_title("Closed-loop mission, Study 2 stack", loc="left")
    b.legend(ncol=3, loc="upper left", fontsize=5.8)
    panel(b, "b")
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig8_heading")


# ---------------------------------------------------------------- self-check

def selfcheck():
    stored = json.loads((OUT / "numbers_sources.json").read_text())
    tex = (OUT / "numbers.tex").read_text()
    J.cache_clear(); runs.cache_clear(); s1.cache_clear(); s1_auc.cache_clear(); s2.cache_clear(); s2_auc.cache_clear()
    for k, v in NUM.items():
        src = stored[k]["source"]
        if src.startswith("results/"):                               # independent re-read of the named key path
            f, path = src[len("results/"):].split(":", 1)
            x = json.loads((ROOT / "results" / f).read_text())
            for p in path.split("/"):
                x = x[int(p)] if isinstance(x, list) else x[p]
        else:
            x = v["fn"]()                                            # derived: recompute from freshly read files
        assert x == stored[k]["value"] or np.isclose(x, stored[k]["value"], rtol=0, atol=1e-12), (k, x, stored[k]["value"])
        assert f"\\newcommand{{\\{k}}}{{{stored[k]['text']}}}" in tex, k
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", (OUT / "main.tex").read_text())) if (OUT / "main.tex").exists() else set()
    missing = {u for u in used if (u.startswith("Sone") or u.startswith("Stwo") or u == "Nseeds") and u not in NUM}
    assert not missing, f"main.tex uses undefined number macros: {missing}"
    m2 = J("mission_v2.json")
    # Study 2 aggregate and per-fault AUC from the run records equal the summary
    for l in m2["arms"]:
        assert np.isclose(s2_agg(l).mean(), m2["aggregate_auc"][l]["mean"], atol=1e-12), l
        for f in FAULTS:
            assert np.isclose(s2_auc(l)[f].mean(), m2["per_fault_auc"][l][f], atol=1e-12), (l, f)
    # (i) Wilson interval recomputed from the records
    k, n = s2_cond("v2_ring", "none", None)
    assert n == 300 and np.allclose(wilson(k, n), m2["H9_i"]["wilson95"], atol=1e-12) and k / n == m2["H9_i"]["success"]
    # (ii) paired bootstrap (10,000, seed 0) recomputed
    for a_, b_, ref in ((s2_agg("v2_ring"), s2_agg("v1"), m2["H9_ii"]["diff"]),
                        (s2_agg("v2_ring"), s2_agg("v1", own=True), m2["H9_ii"]["secondary_v1_own_definition"]["diff"])):
        d = paired_diff(a_, b_)
        assert all(np.isclose(d[q], ref[q], atol=1e-12) for q in ("est", "lo", "hi")), (d, ref)
    for l, ref in m2["ablations_v2_ring_minus_arm"].items():
        d = paired_diff(s2_agg("v2_ring"), s2_agg(l))
        assert all(np.isclose(d[q], ref[q], atol=1e-12) for q in ("est", "lo", "hi")), l
    # (iii) middle-severity success from the records
    for f, mm in m2["H9_iii"]["per_fault"].items():
        k, n = s2_cond("v2_ring", f, mm["severity"])
        assert n == mm["n"] and np.isclose(k / n, mm["success"], atol=1e-12), f
    # arbitration: arm-permutation test recomputed (5,000, seed 0)
    others = [f"v2_{a}" for a in ("ring_static", "vote", "sum", "accumulator", "state_machine")]
    t = arm_permutation_test(s2_agg("v2_ring"), [s2_agg(o) for o in others])
    assert np.isclose(t["p_two_sided"], m2["v2_ring_vs_rivals"]["permutation_test"]["p_two_sided"], atol=1e-12)
    # Study 1: per-fault AUC and the pre-registered ring-minus-best-other bootstrap from the run records
    h1 = J("mission_h1.json")
    for arm, _, _ in S1_ARMS:
        for f in FAULTS:
            assert np.isclose(s1_auc(arm)[f].mean(), h1["per_fault_auc"][arm][f], atol=1e-12), (arm, f)
    agg = [s1_auc(a).mean(axis=1).to_numpy() for a, _, _ in S1_ARMS]
    d = paired_bootstrap(lambda r, *o: r.mean() - max(x.mean() for x in o), *agg)
    assert all(np.isclose(d[q], h1["ring_minus_best_other"][q], atol=1e-12) for q in ("est", "lo", "hi")), d
    for p in FIG.glob("*.pdf"):
        assert p.with_suffix(".png").exists()
    assert len(list(FIG.glob("fig*.pdf"))) == 8
    print(f"self-check OK: {len(NUM)} macros re-read; Study 2 AUCs, Wilson CI, 8 paired bootstraps, permutation test and the "
          f"Study 1 H1 bootstrap recomputed from the run records")


if __name__ == "__main__":
    register()
    write_numbers()
    style()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
    selfcheck()
