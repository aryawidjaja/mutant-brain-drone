"""Adaptive search: a post-study brain module ("adaptive_search", brain.version 2 only; not part of the Study 2 results).

Everything is derived from the brain's own measurements of the remembered target, not from its kind.
Identical equations in app/sim/brain/search.js.

1. Size estimate from the learning flight. For each learning view k the brain knows its horizontal distance d_k to the
   target (path integration along the loop, which starts at the target) and its altitude z_k. It measures the elevation
   of the target's top and base (on the object's vertical centre line) and its angular size alpha_k = atan(D / dist_k)
   (the quantity the eye's detector thresholds, eye.visibility), each quantised to the eye's acuity Q_DEG = 2 deg:
       H_k   = d_k (tan q(e_top) - tan q(e_bot))            object height
       rho_k = hypot(d_k, z_k - H_k / 2)                    range to the object's centre
       D_k   = rho_k tan q(alpha_k)                          object size (diagonal)
   D_hat, H_hat = medians over the views. No learning flight (one photo): prior D_PRIOR, H_PRIOR (a small object).
2. Search geometry. theta_det = DETECT_DEG + DET_MARGIN_DEG; R_det = D_hat / tan(theta_det) is the slant range at
   which the remembered target subtends theta_det. The eye's lower edge is -40 deg; with EL_MARGIN_DEG the usable
   depression is EL_USE = 35 deg. With z_c = H_hat / 2:
       pitch p   = min(p_study, 2 R_det cos(EL_USE))        (p_study: 6 m with a tip, 12 m without)
       altitude h = clip(z_c + (p / 2) tan(EL_USE), ALT_MIN, alt_max)   (alt_max = Study 2 ALT_SEARCH, 3.5 m)
       r_dip     = sqrt(max(R_det^2 - (h - z_c)^2, 0))     horizontal range for inspection dips
   An object at the sweep's lateral half-spacing p / 2 then lies inside the eye's elevation window and, because
   p / 2 <= R_det cos(EL_USE), within R_det (subtends >= theta_det) whenever h is not clamped up to ALT_MIN.
3. Recognition matched to the estimate. Front-end attention window half-width half_m = FRONT_HALF_M min(1, D_hat / D_REF)
   (D_REF = the diagonal of the object the front end and its evidence window were designed on). Evidence window
   self-calibrated after the learning loops from N_CAL + N_CAL unstored look-back views at the search altitude
   (cal_poses): F1 = median familiarity at 1.5-4 m, F0 = median at FAR_FRAC x min(FAR_CAP, D_hat / tan(DETECT_DEG)),
   F1 >= F0 + MIN_SPAN (the train-seed calibration rule of run_v2.calib, per mission and object).
4. Inspection dips (mission.py / mission.js): a visible object slot with 1-s evidence >= DIP_ON (enter; >= DIP_OFF to
   stay) within r_dip while seeking -> altitude view_alt(range): the learning flight's altitude at that horizontal
   range (linear between its first and last pose), clipped to [max(ALT_MIN, H_hat), h] (never below the object's
   estimated top); otherwise h.
Run: uv run python scripts/brain/search.py   (self-check)
"""
import math

import numpy as np

DETECT_DEG, Q_DEG = 4.0, 2.0                 # eye.DETECT_DEG, eye.ACUITY_DEG
DET_MARGIN_DEG, EL_LOW_DEG, EL_MARGIN_DEG = 2.0, -40.0, 5.0
EL_USE = math.radians(-EL_LOW_DEG - EL_MARGIN_DEG)
ALT_MIN = 1.0                                # m: safe minimum (rocks 0.4-1.2 m stay obstacles for the depth ring)
D_PRIOR, H_PRIOR = 1.0, 0.7                  # m: no learning flight -> assume a small object
FRONT_HALF_M, D_REF = 1.1, math.sqrt(2.2 * 2.2 + 1.8 * 1.8 + 1.3 * 1.3)
N_CAL, NEAR, FAR_FRAC, FAR_CAP, MIN_SPAN = 8, (1.5, 4.0), (0.75, 1.0), 20.0, 0.01
DIP_ON, DIP_OFF = 0.15, 0.05


def q(a):
    """Quantise angles (rad) to the eye's acuity."""
    s = math.radians(Q_DEG)
    return s * np.floor(np.asarray(a) / s + 0.5)


def size_estimate(poses, tx, ty, h, diag):
    """poses [B, V, 3] learning-flight eye positions; target base (tx, ty) [B], true height h [B] and diagonal diag [B]
    (used only to render what the eye measures). -> D_hat [B], H_hat [B]."""
    dx, dy = poses[..., 0] - tx[:, None], poses[..., 1] - ty[:, None]
    d = np.sqrt(dx * dx + dy * dy)
    z = poses[..., 2]
    e_top, e_bot = np.arctan2(h[:, None] - z, d), np.arctan2(-z, d)
    zc = z - h[:, None] / 2
    alpha = np.arctan(diag[:, None] / np.sqrt(d * d + zc * zc))
    H = d * (np.tan(q(e_top)) - np.tan(q(e_bot)))
    zr = z - H / 2
    rho = np.sqrt(d * d + zr * zr)
    D = rho * np.tan(q(alpha))
    return np.median(D, 1), np.median(H, 1)


def geometry(D, H, p_study, alt_max):
    """-> altitude h, spiral pitch p, dip range r_dip (all [B])."""
    R = D / math.tan(math.radians(DETECT_DEG + DET_MARGIN_DEG))
    p = np.minimum(p_study, 2 * R * math.cos(EL_USE))
    zc = H / 2
    h = np.clip(zc + p / 2 * math.tan(EL_USE), ALT_MIN, alt_max)
    return h, p, np.sqrt(np.maximum(R * R - (h - zc) ** 2, 0.0))


def half_m(D):
    return FRONT_HALF_M * np.minimum(1.0, D / D_REF)


def cal_poses(tx, ty, D, h):
    """Look-back views (not stored): N_CAL near then N_CAL far, azimuths 2 pi j / N_CAL, altitude h. -> [B, 2 N_CAL, 3]."""
    j = np.arange(N_CAL)
    far = np.minimum(FAR_CAP, D / math.tan(math.radians(DETECT_DEG)))
    r = np.concatenate([np.broadcast_to(NEAR[0] + (NEAR[1] - NEAR[0]) * j / (N_CAL - 1), (len(D), N_CAL)),
                        far[:, None] * (FAR_FRAC[0] + (FAR_FRAC[1] - FAR_FRAC[0]) * j / (N_CAL - 1))], 1)
    a = np.tile(2 * math.pi * j / N_CAL, 2)
    return np.stack([tx[:, None] + r * np.cos(a), ty[:, None] + r * np.sin(a), np.broadcast_to(h[:, None], r.shape)], -1)


def evidence_window(fam):
    """fam [B, 2 N_CAL] (near then far) -> F0, F1 [B]."""
    f1, f0 = np.median(fam[:, :N_CAL], 1), np.median(fam[:, N_CAL:], 1)
    return f0, np.maximum(f1, f0 + MIN_SPAN)


def view_alt(r, d0, z0, d1, z1):
    """The learning flight's altitude at horizontal range r (linear between its first and last pose)."""
    u = np.clip((r - d0) / np.where(d1 > d0, d1 - d0, 1.0), 0.0, 1.0)
    return np.where(d1 > d0, z0 + (z1 - z0) * u, z0)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eye
    assert DETECT_DEG == eye.DETECT_DEG and Q_DEG == eye.ACUITY_DEG and EL_LOW_DEG == eye.EL_TOP - eye.NR * eye.PX_DEG
    dims = {"tent": [2.2, 1.8, 1.3], "backpack": [0.6, 0.4, 0.7]}
    S = eye.Scene.__new__(eye.Scene)
    S.B = 2
    S.obj = np.zeros((2, 1, 17)); S.role = np.ones((2, 1), int)
    for b, k in enumerate(dims):
        S.obj[b, 0, 4:7] = [dims[k][0] / 2, dims[k][1] / 2, dims[k][2]]
    P = eye.learning_poses(S, 10.0)
    diag = eye.obj_diag(S)[:, 0]
    D, H = size_estimate(P, S.obj[:, 0, 0], S.obj[:, 0, 1], S.obj[:, 0, 6], diag)
    assert np.all(np.abs(D / diag - 1) < 0.15) and np.all(np.abs(H / S.obj[:, 0, 6] - 1) < 0.2), (D, diag, H)
    h, p, rd = geometry(D, H, np.full(2, 6.0), 3.5)
    assert h[1] < h[0] <= 3.5 and np.all(h >= ALT_MIN) and np.all(p <= 6.0) and np.all(rd > p / 2)
    zc = H / 2   # the object at the lateral half-spacing is inside the window and big enough
    assert np.all(np.arctan2(h - zc, p / 2) <= EL_USE + 1e-12)
    assert np.all(D / np.hypot(p / 2, h - zc) >= math.tan(math.radians(DETECT_DEG + DET_MARGIN_DEG)))
    tiny = geometry(np.array([0.2]), np.array([0.2]), np.array([6.0]), 3.5)
    assert tiny[1][0] < 6.0 and ALT_MIN <= tiny[0][0] < h[1]          # very small objects: tighter spiral, lower
    assert abs(half_m(np.array([D_REF * 2]))[0] - FRONT_HALF_M) < 1e-12 and half_m(D)[1] < 0.5
    C = cal_poses(np.zeros(2), np.zeros(2), D, h)
    assert C.shape == (2, 2 * N_CAL, 3) and np.allclose(np.hypot(C[:, 0, 0], C[:, 0, 1]), NEAR[0])
    f0, f1 = evidence_window(np.array([[0.4] * N_CAL + [0.3] * N_CAL, [0.2] * N_CAL + [0.2] * N_CAL]))
    assert np.allclose(f0, [0.3, 0.2]) and np.allclose(f1, [0.4, 0.21])
    assert np.allclose(view_alt(np.array([0.0, 4.5, 9.0]), 2.0, 1.0, 7.0, 3.5), [1.0, 2.25, 3.5])
    assert view_alt(np.array([3.0]), 4.0, 2.5, 4.0, 2.5)[0] == 2.5
    print(f"search self-check OK: D_hat {np.round(D, 3)} (true {np.round(diag, 3)}), H_hat {np.round(H, 3)}, "
          f"altitude {np.round(h, 2)} m, pitch {np.round(p, 2)} m, dip range {np.round(rd, 1)} m, half_m {np.round(half_m(D), 3)}")
