"""Crazyflie 2.0 quadrotor: batched 6-DOF rigid body (RK4, quaternion) + cascade PID controller.

Parameters and their sources: results/vehicle_params.json (Menagerie cf2 mass/inertia, Förster 2015 motor
maps, lag and drag, firmware arm length and mixer). app/sim/quad.js is a line-by-line port (batch 1).

Frames: world ENU (x east, y north, z up); body FLU (x forward, y left, z up); quat (w,x,y,z) body->world.
Motors (firmware layout): M1 front-right CCW, M2 back-right CW, M3 back-left CCW, M4 front-left CW.

Quad.step(cmd, wind, dt), in this order (same in JS):
  1. motor thrust lag, exact ZOH: f <- a f + (1-a) f_ss(cmd), a = exp(-dt/tau), f_ss = Förster PWM->thrust map
  2. RK4 of (pos, vel, quat, omega) with motor thrusts and wind held over the step; forces:
       gravity, thrust along body z, Förster drag fa = Kaero * sum(rotor speed) * R^T (vel - wind)
     torques: X-layout thrust moments, rotor drag torque thrust2torque*f, gyroscopic omega x I omega
  3. quaternion renormalised
  4. ground: z < 0 -> z = 0, vz = max(vz, 0), horizontal velocity zeroed (crude friction)
Simplifications: no ground effect, blade flapping, battery sag or motor latency beyond the fitted lag.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAMS_JSON = ROOT / "results" / "vehicle_params.json"
PARAMS_JS = ROOT / "app" / "sim" / "vehicle_params.js"


def load_params(path=PARAMS_JSON):
    """Flat {name: value} from vehicle_params.json; derived constants recomputed from measured ones."""
    P = {k: v["value"] for k, v in json.loads(Path(path).read_text())["params"].items()}
    P["motor_tau"] = -P["motor_pole_dt"] / np.log(P["motor_pole"])
    P["f_max"] = P["thrust_a2"] * P["cmd_max"] ** 2 + P["thrust_a1"] * P["cmd_max"] + P["thrust_a0"]
    return P


P = load_params()

# Controller gains (SI; our tuning, structure from the Crazyflie firmware cascade). Same object in quad.js.
GAINS = {
    "kp_alt": 2.0,          # 1/s       altitude error -> vertical velocity setpoint
    "vz_max": 1.5,          # m/s
    "kp_vz": 6.0, "ki_vz": 3.0, "i_vz_max": 2.0,       # -> vertical acceleration (m/s^2)
    "kp_vxy": 2.5, "ki_vxy": 0.6, "i_vxy_max": 2.0,    # -> horizontal acceleration (m/s^2)
    "tilt_max": 0.35,       # rad (20 deg; firmware default roll/pitch limit is also 20 deg)
    "kp_att": 9.0,          # 1/s       roll/pitch angle error -> body rate setpoint
    "kp_yaw": 4.0,          # 1/s       yaw error -> yaw rate setpoint
    "rate_max": 6.0,        # rad/s     roll/pitch rate setpoint limit
    "yaw_rate_max": 3.0,    # rad/s
    "kp_rate": 60.0, "ki_rate": 20.0, "kd_rate": 1.5, "i_rate_max": 3.0,          # roll/pitch (1/s, ...)
    "kp_rate_yaw": 25.0, "ki_rate_yaw": 5.0, "kd_rate_yaw": 0.0, "i_rate_yaw_max": 3.0,
}


def thrust_of_cmd(u, P=P):
    """Steady thrust per motor (N) for normalised command u in [0,1] (Förster quadratic PWM map)."""
    c = u * P["cmd_max"]
    return P["thrust_a2"] * c * c + P["thrust_a1"] * c + P["thrust_a0"]


def cmd_of_thrust(f, P=P):
    """Inverse of thrust_of_cmd, clipped to [0,1]."""
    a2, a1, a0 = P["thrust_a2"], P["thrust_a1"], P["thrust_a0"]
    disc = a1 * a1 + 4.0 * a2 * np.maximum(f - a0, 0.0)
    return np.clip((-a1 + np.sqrt(disc)) / (2.0 * a2) / P["cmd_max"], 0.0, 1.0)


def quat_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def quat_to_euler(q):
    """ZYX Euler (roll, pitch, yaw) in rad, [B,3]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], 1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motor_wrench(f, P=P):
    """Per-motor thrusts f [B,4] -> (total thrust [B], body torque [B,3]). Firmware X layout."""
    r = P["arm_length"] / np.sqrt(2.0)
    T = f[:, 0] + f[:, 1] + f[:, 2] + f[:, 3]
    tx = r * (-f[:, 0] - f[:, 1] + f[:, 2] + f[:, 3])
    ty = r * (-f[:, 0] + f[:, 1] + f[:, 2] - f[:, 3])
    tz = P["thrust2torque"] * (-f[:, 0] + f[:, 1] - f[:, 2] + f[:, 3])
    return T, np.stack([tx, ty, tz], 1)


class Quad:
    def __init__(self, B, params=P, pos0=None, yaw0=None):
        self.B, self.P = B, params
        self.I = np.array([params["Ixx"], params["Iyy"], params["Izz"]])
        self.pos = np.zeros((B, 3)) if pos0 is None else np.broadcast_to(np.asarray(pos0, float), (B, 3)).copy()
        self.vel = np.zeros((B, 3))
        yaw0 = np.zeros(B) if yaw0 is None else np.broadcast_to(np.asarray(yaw0, float), (B,)).copy()
        self.quat = np.stack([np.cos(yaw0 / 2), 0 * yaw0, 0 * yaw0, np.sin(yaw0 / 2)], 1)
        self.omega = np.zeros((B, 3))
        self.motor = np.full((B, 4), params["mass"] * params["g"] / 4.0)  # start at hover thrust

    def euler(self):
        return quat_to_euler(self.quat)

    def update_motors(self, cmd, dt):
        a = np.exp(-dt / self.P["motor_tau"])
        self.motor = a * self.motor + (1.0 - a) * thrust_of_cmd(np.clip(cmd, 0.0, 1.0), self.P)

    def _deriv(self, vel, q, om, T, tau, rotsum, wind):
        P, I = self.P, self.I
        R = quat_to_R(q)
        va = np.einsum("bji,bj->bi", R, vel - wind)  # air velocity in body frame (R^T v)
        fb = np.stack([P["kaero_xy"] * rotsum * va[:, 0], P["kaero_xy"] * rotsum * va[:, 1],
                       P["kaero_z"] * rotsum * va[:, 2] + T], 1)
        acc = np.einsum("bij,bj->bi", R, fb) / P["mass"]
        acc[:, 2] -= P["g"]
        Iw = I * om
        omd = (tau - np.cross(om, Iw)) / I
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        p, qq, r = om[:, 0], om[:, 1], om[:, 2]
        qd = 0.5 * np.stack([-x * p - y * qq - z * r, w * p + y * r - z * qq, w * qq + z * p - x * r, w * r + x * qq - y * p], 1)
        return vel, acc, qd, omd

    def step(self, cmd, wind, dt):
        self.update_motors(cmd, dt)
        T, tau = motor_wrench(self.motor, self.P)
        rotsum = np.sum(self.P["rotor_b1"] * cmd_of_thrust(self.motor, self.P) * self.P["cmd_max"] + self.P["rotor_b0"], 1)
        wind = np.broadcast_to(np.asarray(wind, float), (self.B, 3))
        s0 = (self.pos, self.vel, self.quat, self.omega)

        def f(s):
            return self._deriv(s[1], s[2], s[3], T, tau, rotsum, wind)

        def add(s, k, h):
            return tuple(a + h * b for a, b in zip(s, k))

        k1 = f(s0)
        k2 = f(add(s0, k1, dt / 2))
        k3 = f(add(s0, k2, dt / 2))
        k4 = f(add(s0, k3, dt))
        self.pos, self.vel, self.quat, self.omega = (
            a + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s0, k1, k2, k3, k4))
        self.quat = self.quat / np.linalg.norm(self.quat, axis=1, keepdims=True)
        g = self.pos[:, 2] < 0.0
        if g.any():
            self.pos[g, 2] = 0.0
            self.vel[g, 2] = np.maximum(self.vel[g, 2], 0.0)
            self.vel[g, :2] = 0.0


class Controller:
    """Cascade PID: altitude -> vz; vel_xy -> tilt; attitude -> body rate -> torque -> mixer -> motor cmd [0,1]."""

    def __init__(self, B, gains=None, params=P):
        self.B, self.P, self.G = B, params, dict(GAINS, **(gains or {}))
        self.I = np.array([params["Ixx"], params["Iyy"], params["Izz"]])
        self.i_vz = np.zeros(B)
        self.i_vxy = np.zeros((B, 2))
        self.i_rate = np.zeros((B, 3))
        self.om_prev = None

    def __call__(self, quad, sp, dt):
        G, P = self.G, self.P
        m, g = P["mass"], P["g"]
        eul = quad.euler()
        roll, pitch, yaw = eul[:, 0], eul[:, 1], eul[:, 2]
        # altitude -> vertical velocity -> vertical acceleration (PI)
        vz_sp = np.clip(G["kp_alt"] * (np.asarray(sp["alt"], float) - quad.pos[:, 2]), -G["vz_max"], G["vz_max"])
        e_vz = vz_sp - quad.vel[:, 2]
        self.i_vz = np.clip(self.i_vz + e_vz * dt, -G["i_vz_max"], G["i_vz_max"])
        az = G["kp_vz"] * e_vz + G["ki_vz"] * self.i_vz
        # horizontal velocity -> horizontal acceleration (PI) -> tilt in the yaw frame
        e_v = np.asarray(sp["vel_xy"], float) - quad.vel[:, :2]
        self.i_vxy = np.clip(self.i_vxy + e_v * dt, -G["i_vxy_max"], G["i_vxy_max"])
        a_xy = G["kp_vxy"] * e_v + G["ki_vxy"] * self.i_vxy
        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_b = cy * a_xy[:, 0] + sy * a_xy[:, 1]
        ay_b = -sy * a_xy[:, 0] + cy * a_xy[:, 1]
        gz = np.maximum(g + az, 0.2 * g)
        pitch_sp = np.clip(np.arctan2(ax_b, gz), -G["tilt_max"], G["tilt_max"])
        roll_sp = np.clip(np.arctan2(-ay_b * np.cos(pitch_sp), gz), -G["tilt_max"], G["tilt_max"])
        T = m * gz / np.maximum(np.cos(roll) * np.cos(pitch), 0.5)
        # attitude (P) -> body rate setpoint
        rate_sp = np.stack([
            np.clip(G["kp_att"] * (roll_sp - roll), -G["rate_max"], G["rate_max"]),
            np.clip(G["kp_att"] * (pitch_sp - pitch), -G["rate_max"], G["rate_max"]),
            np.clip(G["kp_yaw"] * wrap(np.asarray(sp["yaw"], float) - yaw), -G["yaw_rate_max"], G["yaw_rate_max"])], 1)
        # body rate PID (derivative on measurement) -> torque
        om = quad.omega
        if self.om_prev is None:
            self.om_prev = om.copy()
        e_r = rate_sp - om
        imax = np.array([G["i_rate_max"], G["i_rate_max"], G["i_rate_yaw_max"]])
        self.i_rate = np.clip(self.i_rate + e_r * dt, -imax, imax)
        kp = np.array([G["kp_rate"], G["kp_rate"], G["kp_rate_yaw"]])
        ki = np.array([G["ki_rate"], G["ki_rate"], G["ki_rate_yaw"]])
        kd = np.array([G["kd_rate"], G["kd_rate"], G["kd_rate_yaw"]])
        alpha = kp * e_r + ki * self.i_rate - kd * (om - self.om_prev) / dt
        self.om_prev = om.copy()
        tau = self.I * alpha
        # mixer (firmware powerDistributionForceTorque) + cap (firmware powerDistributionCap)
        r = P["arm_length"] / np.sqrt(2.0)
        tp, rp, pp, yp = 0.25 * T, 0.25 / r * tau[:, 0], 0.25 / r * tau[:, 1], 0.25 * tau[:, 2] / P["thrust2torque"]
        f = np.stack([tp - rp - pp - yp, tp - rp + pp + yp, tp + rp + pp - yp, tp + rp - pp + yp], 1)
        f = f - np.maximum(f.max(1, keepdims=True) - P["f_max"], 0.0)
        return cmd_of_thrust(np.clip(f, 0.0, P["f_max"]), P)


def export_js(path=PARAMS_JS):
    """Write app/sim/vehicle_params.js (static ES module): P (float64 repr round-trips exactly), GAINS, SOURCES."""
    src = json.loads(PARAMS_JSON.read_text())
    body = {"P": load_params(), "GAINS": GAINS, "SOURCES": src["params"]}  # P carries the recomputed derived values
    path.write_text("// GENERATED by scripts/quad.py from results/vehicle_params.json. Do not edit.\n"
                    + "".join(f"export const {k} = {json.dumps(v, indent=1, ensure_ascii=False)};\n" for k, v in body.items()))


def fly(B, sp_fn, T_s, dt=0.002, wind_fn=None, pos0=(0, 0, 1.0), yaw0=0.0, gains=None):
    """Closed-loop run; returns dict of arrays [N, B, ...] logged every step."""
    q, c = Quad(B, pos0=pos0, yaw0=yaw0), Controller(B, gains)
    log = {k: [] for k in ("t", "pos", "eul", "motor", "cmd")}
    for k in range(int(round(T_s / dt))):
        t = k * dt
        cmd = c(q, sp_fn(t), dt)
        q.step(cmd, np.zeros((B, 3)) if wind_fn is None else wind_fn(t), dt)
        for name, v in (("t", t + dt), ("pos", q.pos.copy()), ("eul", q.euler()), ("motor", q.motor.copy()), ("cmd", cmd)):
            log[name].append(v)
    return {k: np.array(v) for k, v in log.items()}


def settle_time(t, y, target, tol):
    """First time after which |y - target| stays within tol (inf if never)."""
    bad = np.where(np.abs(y - target) > tol)[0]
    return 0.0 if len(bad) == 0 else (t[bad[-1] + 1] if bad[-1] + 1 < len(t) else np.inf)


def mujoco_check(T_s=3.0, dt=0.002):
    """Rigid-body check against MuJoCo (Menagerie cf2.xml, its own RK4): both run the same controller and motor
    model; MuJoCo gets total thrust and body moments through cf2's lumped actuators (ctrl limits removed, since the
    README calls them arbitrary) and our drag through xfrc_applied (held over the step; MuJoCo's own fluid model off).
    __main__ --mujoco writes results/quad_mujoco_check.json (with drag, and with drag off to isolate the rigid body)."""
    import mujoco
    from provenance import stamp
    m = mujoco.MjModel.from_xml_path(str(ROOT / "vendor/mujoco_menagerie/bitcraze_crazyflie_2/cf2.xml"))
    assert abs(m.body_mass[1] - P["mass"]) < 1e-12 and np.allclose(m.body_inertia[1], [P["Ixx"], P["Iyy"], P["Izz"]])
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    m.opt.density, m.opt.viscosity, m.opt.timestep = 0.0, 0.0, dt
    m.actuator_ctrllimited[:] = 0
    gear = m.actuator_gear[1:4, 3:6].diagonal()  # (-1e-5, -1e-5, -1e-5)
    zero2 = np.zeros((1, 2))
    cases = {
        "hover": dict(sp=lambda t: {"vel_xy": zero2, "alt": np.ones(1), "yaw": np.zeros(1)}),
        "yaw_step_1rad": dict(sp=lambda t: {"vel_xy": zero2, "alt": np.ones(1), "yaw": np.full(1, 1.0 * (t >= 0.5))}),
        "attitude_step_via_vel_x_1mps": dict(sp=lambda t: {"vel_xy": np.array([[1.0 * (t >= 0.5), 0]]), "alt": np.ones(1), "yaw": np.zeros(1)}),
        "wind_2mps_drag": dict(sp=lambda t: {"vel_xy": zero2, "alt": np.ones(1), "yaw": np.zeros(1)}, wind=np.array([[2.0, -1.0, 0.0]])),
        "open_loop_roll_yaw_pulse_0p8s": dict(open_loop=True, T=0.8),
    }
    hov = float(cmd_of_thrust(np.array(P["mass"] * P["g"] / 4)))

    def open_cmd(t):  # 0.1 s differential pulse from hover: roll + yaw, then back to hover command
        return np.array([[hov, hov, hov, hov]]) + (0.0 <= t < 0.1) * np.array([[-0.005, -0.002, 0.005, 0.002]])

    out = {}
    for name, cs in cases.items():
        wind = cs.get("wind", np.zeros((1, 3)))
        ref, c_ref = Quad(1, pos0=(0, 0, 1.0)), Controller(1)
        sh, c_mj = Quad(1, pos0=(0, 0, 1.0)), Controller(1)
        d = mujoco.MjData(m)
        d.qpos[:3], d.qpos[3:7] = [0, 0, 1.0], [1, 0, 0, 0]
        err_p, err_q, err_w = 0.0, 0.0, 0.0
        for k in range(int(round(cs.get("T", T_s) / dt))):
            t = k * dt
            cmd = open_cmd(t) if cs.get("open_loop") else c_ref(ref, cs["sp"](t), dt)
            ref.step(cmd, wind, dt)
            sh.pos, sh.vel = d.qpos[None, :3].copy(), d.qvel[None, :3].copy()
            sh.quat, sh.omega = d.qpos[None, 3:7].copy(), d.qvel[None, 3:6].copy()  # free joint: omega in body frame
            cmd = open_cmd(t) if cs.get("open_loop") else c_mj(sh, cs["sp"](t), dt)
            sh.update_motors(cmd, dt)
            T, tau = motor_wrench(sh.motor)
            rotsum = np.sum(P["rotor_b1"] * cmd_of_thrust(sh.motor) * P["cmd_max"] + P["rotor_b0"], 1)
            R = quat_to_R(sh.quat)[0]
            va = R.T @ (sh.vel[0] - wind[0])
            d.xfrc_applied[1, :3] = R @ (np.array([P["kaero_xy"], P["kaero_xy"], P["kaero_z"]]) * rotsum[0] * va)
            d.ctrl[:] = [T[0], *(tau[0] / gear)]
            mujoco.mj_step(m, d)
            err_p = max(err_p, float(np.abs(d.qpos[:3] - ref.pos[0]).max()))
            err_q = max(err_q, float(1 - abs(np.dot(d.qpos[3:7], ref.quat[0]))))
            err_w = max(err_w, float(np.abs(d.qvel[3:6] - ref.omega[0]).max()))
        out[name] = {"max_pos_dev_m": err_p, "max_quat_dev_1_minus_abs_dot": err_q, "max_omega_dev_radps": err_w,
                     "final_pos_numpy": ref.pos[0].tolist(), "final_pos_mujoco": d.qpos[:3].tolist(),
                     "final_euler_deg_numpy": np.degrees(ref.euler()[0]).tolist()}
    res = {"what": "numpy RK4 rigid body vs MuJoCo 3 (Menagerie cf2.xml, RK4) with identical motor model and controller",
           "duration_s": T_s, "dt_s": dt, "mujoco": mujoco.__version__, "cases": out,
           "max_pos_dev_m": max(v["max_pos_dev_m"] for v in out.values()),
           "provenance": stamp(seed=None, param_source="results/vehicle_params.json + vendor/mujoco_menagerie@0059d43 cf2.xml")}
    return res


if __name__ == "__main__":
    import sys
    import time
    if "--mujoco" in sys.argv:
        kx, kz = P["kaero_xy"], P["kaero_z"]
        P["kaero_xy"] = P["kaero_z"] = 0.0
        nodrag = mujoco_check()
        P["kaero_xy"], P["kaero_z"] = kx, kz
        res = mujoco_check()
        res["drag_off"] = {"max_pos_dev_m": nodrag["max_pos_dev_m"], "cases": nodrag["cases"],
                           "note": "Förster drag set to 0 in both: isolates the rigid-body equations. With drag on, the residual comes from MuJoCo's xfrc_applied being held over each RK4 step while our drag is re-evaluated at every RK4 stage."}
        (ROOT / "results" / "quad_mujoco_check.json").write_text(json.dumps(res, indent=1))
        for k in res["cases"]:
            print(f"{k:32s} drag on {res['cases'][k]['max_pos_dev_m']:.2e} m   drag off {nodrag['cases'][k]['max_pos_dev_m']:.2e} m")
        assert nodrag["max_pos_dev_m"] < 1e-9, nodrag["max_pos_dev_m"]
        assert res["max_pos_dev_m"] < 1e-3, res["max_pos_dev_m"]
        moved = res["cases"]["attitude_step_via_vel_x_1mps"]["final_pos_numpy"][0]
        assert moved > 1.0, moved  # the check is not vacuous
        print("mujoco check OK, max pos dev", res["max_pos_dev_m"])
        sys.exit(0)

    from dryden import Dryden
    from provenance import stamp

    raw = json.loads(PARAMS_JSON.read_text())["params"]
    assert abs(P["motor_tau"] - raw["motor_tau"]["value"]) / P["motor_tau"] < 1e-3, P["motor_tau"]
    assert abs(P["f_max"] - raw["f_max"]["value"]) / P["f_max"] < 1e-3, P["f_max"]
    assert all({"value", "unit", "source", "quote"} <= set(v) for v in raw.values())
    assert np.allclose(thrust_of_cmd(cmd_of_thrust(np.array([0.01, 0.0662, 0.15]))), [0.01, 0.0662, 0.15])
    hov = 0.25 * P["mass"] * P["g"]
    assert hov < 0.5 * P["f_max"]
    out, t0 = {"params_derived": {"motor_tau_s": P["motor_tau"], "f_max_N": P["f_max"], "hover_thrust_per_motor_N": hov,
                                  "hover_cmd": float(cmd_of_thrust(np.array(hov)))}, "gains": GAINS}, time.time()

    # 1. hover, no wind, 10 s (pre-registered gate: within 0.05 m)
    r = fly(4, lambda t: {"vel_xy": np.zeros((4, 2)), "alt": np.full(4, 1.0), "yaw": np.zeros(4)}, 10.0)
    hover_err = float(np.abs(r["pos"] - np.array([0, 0, 1.0])).max())
    assert hover_err < 0.05, hover_err
    out["hover_10s_max_pos_err_m"] = hover_err  # exact equilibrium (motors start at hover thrust)
    # hover recovery from a 10 deg roll + 0.5 m/s kick (not a gate; shows the loops pull back)
    q, c = Quad(1, pos0=(0, 0, 1.0)), Controller(1)
    q.quat = np.array([[np.cos(0.0873), np.sin(0.0873), 0, 0]])
    q.vel[:] = [0.5, 0, 0]
    for _ in range(5000):
        q.step(c(q, {"vel_xy": np.zeros((1, 2)), "alt": np.ones(1), "yaw": np.zeros(1)}, 0.002), np.zeros((1, 3)), 0.002)
    out["kick_recovery_after_10s"] = {"speed_mps": float(np.linalg.norm(q.vel)), "alt_err_m": float(abs(q.pos[0, 2] - 1)),
                                      "tilt_deg": float(np.degrees(np.abs(q.euler()[0, :2]).max()))}
    assert out["kick_recovery_after_10s"]["speed_mps"] < 0.01 and out["kick_recovery_after_10s"]["alt_err_m"] < 0.01

    # 2. step responses (batch of 3 different steps at once)
    def sp_steps(t):
        on = t >= 1.0
        return {"vel_xy": np.array([[1.0 * on, 0], [0, 0], [0, 0]]), "alt": np.array([1.0, 1.0 + 0.5 * on, 1.0]),
                "yaw": np.array([0, 0, 1.0 * on])}
    r = fly(3, sp_steps, 6.0)
    t = r["t"]
    vx = np.gradient(r["pos"][:, 0, 0], t)
    ts = {"vel_x_1mps_settle_s": settle_time(t, vx, 1.0, 0.1) - 1.0,
          "alt_0p5m_settle_s": settle_time(t, r["pos"][:, 1, 2], 1.5, 0.05) - 1.0,
          "yaw_1rad_settle_s": settle_time(t, r["eul"][:, 2, 2], 1.0, 0.05) - 1.0,
          "vel_x_overshoot": float(vx.max() - 1.0), "alt_overshoot_m": float(r["pos"][:, 1, 2].max() - 1.5),
          "yaw_overshoot_rad": float(r["eul"][:, 2, 2].max() - 1.0),
          "max_tilt_deg": float(np.degrees(np.abs(r["eul"][:, :, :2]).max()))}
    ts = {k: float(v) for k, v in ts.items()}
    assert ts["vel_x_1mps_settle_s"] < 3.0 and ts["alt_0p5m_settle_s"] < 3.0 and ts["yaw_1rad_settle_s"] < 3.0, ts
    assert ts["vel_x_overshoot"] < 0.3 and ts["alt_overshoot_m"] < 0.15 and ts["yaw_overshoot_rad"] < 0.2, ts
    out["steps"] = ts

    # 3. batch shapes
    q, c = Quad(5, pos0=(0, 0, 2.0), yaw0=np.linspace(0, 1, 5)), Controller(5)
    cmd = c(q, {"vel_xy": np.zeros((5, 2)), "alt": np.full(5, 2.0), "yaw": np.zeros(5)}, 0.002)
    q.step(cmd, np.zeros((5, 3)), 0.002)
    assert cmd.shape == (5, 4) and q.pos.shape == (5, 3) and q.quat.shape == (5, 4) and q.euler().shape == (5, 3)
    assert ((cmd >= 0) & (cmd <= 1)).all()

    # 4. gusts: severe Dryden + 3 m/s mean wind, hover at 3 m for 20 s: no divergence
    dry = Dryden(8, np.arange(8), "severe", (3.0, 0.0, 0.0), alt_m=3.0)
    r = fly(8, lambda t: {"vel_xy": np.zeros((8, 2)), "alt": np.full(8, 3.0), "yaw": np.zeros(8)}, 20.0,
            wind_fn=lambda t: dry.step(0.002), pos0=(0, 0, 3.0))
    gust = {"severe_mean3_max_horiz_dev_m": float(np.linalg.norm(r["pos"][:, :, :2], axis=2).max()),
            "severe_mean3_max_alt_dev_m": float(np.abs(r["pos"][:, :, 2] - 3.0).max()),
            "severe_mean3_max_tilt_deg": float(np.degrees(np.abs(r["eul"][:, :, :2]).max())),
            "motor_saturation_frac": float((r["cmd"] >= 0.999).mean())}
    assert np.isfinite(r["pos"]).all() and gust["severe_mean3_max_horiz_dev_m"] < 10.0 and gust["severe_mean3_max_alt_dev_m"] < 2.0, gust
    dry = Dryden(8, np.arange(8), "light", (0.0, 0.0, 0.0), alt_m=1.0)
    r = fly(8, lambda t: {"vel_xy": np.zeros((8, 2)), "alt": np.full(8, 1.0), "yaw": np.zeros(8)}, 10.0,
            wind_fn=lambda t: dry.step(0.002))
    gust["light_hover_10s_max_horiz_dev_m"] = float(np.linalg.norm(r["pos"][:, :, :2], axis=2).max())  # info only
    gust["light_hover_10s_max_alt_dev_m"] = float(np.abs(r["pos"][:, :, 2] - 1.0).max())
    out["gust"] = gust

    # 5. ground clamp: commanded below ground, stays at z >= 0
    r = fly(1, lambda t: {"vel_xy": np.zeros((1, 2)), "alt": np.full(1, -1.0), "yaw": np.zeros(1)}, 3.0)
    assert r["pos"][:, 0, 2].min() >= 0.0 and r["pos"][-1, 0, 2] == 0.0

    # timing: physics + controller steps per second at B=1 and B=300
    for B in (1, 300):
        q, c = Quad(B, pos0=(0, 0, 1.0)), Controller(B)
        spB = {"vel_xy": np.zeros((B, 2)), "alt": np.ones(B), "yaw": np.zeros(B)}
        t1 = time.time()
        for _ in range(500):
            q.step(c(q, spB, 0.002), np.zeros((B, 3)), 0.002)
        out[f"steps_per_s_B{B}"] = 500 / (time.time() - t1)

    export_js()
    (ROOT / "results" / "quad_selfcheck.json").write_text(json.dumps(
        dict(out, provenance=stamp(seed="0-7 (Dryden gust test)", param_source="results/vehicle_params.json")), indent=1))
    print(json.dumps(out, indent=1))
    print("quad OK")
