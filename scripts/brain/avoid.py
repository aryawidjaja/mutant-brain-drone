"""Obstacle reflexes: optic-flow balancing and a looming reflex (engineered from published principles; flyvis has
no LPLC2/HS cells, so this is not a model of identified neurons). Identical in app/sim/brain/avoid.js.

Input is geometric optic flow from the eye's depth ring plus noise (eye.depth_ring; "geometric optic flow from
ray-casts plus noise"): for column azimuth phi (body frame) at range d, translational flow magnitude
  F(phi) = |v_f sin(phi) - v_s cos(phi)| / d + noise      (rad/s),  v = body velocity estimate (forward, left)
and frontal expansion (inverse time-to-contact) E(phi) = max(0, v_f cos(phi) + v_s sin(phi)) / d.
  balance:  yaw_offset_bal = -K_BAL * (F_left - F_right) / (F_left + F_right + eps) * tanh(max(F_l, F_r) / F_REF)
            (left = phi in [20, 90] deg, right = [-90, -20] deg): turn away from the side that streams faster.
  looming:  loom = max over |phi| <= 30 deg of E. Above 1/TAU_LOOM the reflex turns toward the side with more
            free space in the front quadrants (+/- 90 deg) and slows the drone.
Dead eye columns see no flow (they contribute 0).
These reflexes are added after the arbiter, at the "descending neuron" level: Rayshubskiy et al. 2025 (eLife,
https://elifesciences.org/articles/102230) find descending neurons that "integrate internally directed steering
signals from the head direction system with stimulus-directed steering signals from multimodal sensory pathways".
"""
import numpy as np

from eye import col_az, NC

K_BAL = 0.6            # rad
F_REF = 0.5            # rad/s
TAU_LOOM = 2.0         # s
K_LOOM = 1.2           # rad
SIGMA_FLOW = 0.05      # rad/s
R_DRONE = 0.15         # m, collision radius (Crazyflie motor-to-motor 0.092 m plus rotor margin; ours)

PHI = col_az()                                      # body-relative column azimuths
LEFT = (PHI >= np.radians(20)) & (PHI <= np.radians(90))
RIGHT = (PHI <= np.radians(-20)) & (PHI >= np.radians(-90))
FRONT = np.abs(PHI) <= np.radians(30)
FL = (PHI > 0) & (PHI <= np.radians(90))
FR = (PHI < 0) & (PHI >= np.radians(-90))


def reflex(depth, v_body, noise, dead=None):
    """depth [B,NC] m, v_body [B,2] (forward, left) m/s, noise [B,NC] standard normals, dead [B,NC] bool or None.
    -> yaw_offset [B] rad, speed_factor [B] in [0.3, 1], loom [B] 1/s."""
    vf, vs = v_body[:, :1], v_body[:, 1:]
    F = np.abs(vf * np.sin(PHI) - vs * np.cos(PHI)) / depth + SIGMA_FLOW * noise
    E = np.maximum(0.0, vf * np.cos(PHI) + vs * np.sin(PHI)) / depth
    if dead is not None:
        F, E = np.where(dead, 0.0, F), np.where(dead, 0.0, E)
        depth = np.where(dead, 20.0, depth)               # a blind column looks open
    fl, fr = F[:, LEFT].mean(-1), F[:, RIGHT].mean(-1)
    bal = -K_BAL * (fl - fr) / (fl + fr + 1e-6) * np.tanh(np.maximum(fl, fr) / F_REF)
    loom = E[:, FRONT].max(-1)
    excess = np.clip(loom * TAU_LOOM - 1.0, 0.0, 1.0)
    side = np.where(depth[:, FL].mean(-1) >= depth[:, FR].mean(-1), 1.0, -1.0)
    return bal + side * K_LOOM * excess, 1.0 - 0.7 * excess, loom


def collided(sc, pos):
    """True where the drone (sphere R_DRONE) is inside any obstacle. sc: eye.Scene, pos [B,3]."""
    c = sc.cyl
    dc = np.hypot(pos[:, None, 0] - c[..., 0], pos[:, None, 1] - c[..., 1])
    hit_c = (dc < c[..., 2] + R_DRONE) & (pos[:, None, 2] < c[..., 3] + R_DRONE)
    b = sc.box
    hit_b = ((np.abs(pos[:, None, 0] - b[..., 0]) < b[..., 2] + R_DRONE) & (np.abs(pos[:, None, 1] - b[..., 1]) < b[..., 3] + R_DRONE)
             & (pos[:, None, 2] < b[..., 4] + R_DRONE))
    return hit_c.any(-1) | hit_b.any(-1)


if __name__ == "__main__":
    B = 3
    open_ = np.full((B, NC), 20.0)
    z = np.zeros((B, NC))
    v = np.tile([2.0, 0.0], (B, 1))
    yo, sf, lm = reflex(open_, v, z)
    assert np.allclose(yo, 0) and np.allclose(sf, 1) and np.all(lm < 1 / TAU_LOOM)
    # wall close on the left -> turn right (negative yaw offset)
    d = open_.copy(); d[:, LEFT] = 1.5
    yo, _, _ = reflex(d, v, z)
    assert np.all(yo < -0.1), yo
    # obstacle dead ahead at 2 m, more room on the right -> loom high, slow down, turn right
    d = open_.copy(); d[:, FRONT] = 2.0; d[:, FL] = np.minimum(d[:, FL], 4.0)
    yo, sf, lm = reflex(d, v, z)
    assert np.all(lm > 1 / TAU_LOOM) and np.all(sf < 1) and np.all(yo < -0.5), (yo, sf, lm)
    # dead columns on the left hide the wall, and the empty flow there pulls the drone toward the blind side
    d = open_.copy(); d[:, LEFT] = 1.5
    yo, _, _ = reflex(d, v, z, dead=np.broadcast_to(LEFT, (B, NC)))
    assert np.all(yo > 0), yo
    print("avoid self-check OK")
