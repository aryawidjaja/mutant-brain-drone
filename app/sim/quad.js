// Crazyflie 2.0 quadrotor + cascade PID. Line-by-line port of scripts/quad.py; equations,
// step order and frames are documented there. Parity: scripts/parity_check.py (results/parity.json).
import { P as P0, GAINS as G0 } from './vehicle_params.js';

export const P = P0, GAINS = G0;

export function thrustOfCmd(u, P = P0) {
  const c = u * P.cmd_max;
  return P.thrust_a2 * c * c + P.thrust_a1 * c + P.thrust_a0;
}

export function cmdOfThrust(f, P = P0) {
  const disc = P.thrust_a1 * P.thrust_a1 + 4.0 * P.thrust_a2 * Math.max(f - P.thrust_a0, 0.0);
  return Math.min(Math.max((-P.thrust_a1 + Math.sqrt(disc)) / (2.0 * P.thrust_a2) / P.cmd_max, 0.0), 1.0);
}

export function quatToR(q) {
  const [w, x, y, z] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
          2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
          2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}

export function quatToEuler(q) {
  const [w, x, y, z] = q;
  return [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
          Math.asin(Math.min(Math.max(2 * (w * y - z * x), -1.0), 1.0)),
          Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
}

export function wrap(a) {
  const b = 2 * Math.PI, m = (a + Math.PI) % b;
  return (m < 0 ? m + b : m) - Math.PI;
}

export function motorWrench(f, P = P0) {
  const r = P.arm_length / Math.sqrt(2.0);
  return [f[0] + f[1] + f[2] + f[3],
          [r * (-f[0] - f[1] + f[2] + f[3]), r * (-f[0] + f[1] + f[2] - f[3]), P.thrust2torque * (-f[0] + f[1] - f[2] + f[3])]];
}

export class Quad {
  constructor(params = P0, pos0 = [0, 0, 0], yaw0 = 0) {
    this.P = params;
    this.I = [params.Ixx, params.Iyy, params.Izz];
    this.pos = Float64Array.from(pos0);
    this.vel = new Float64Array(3);
    this.quat = Float64Array.from([Math.cos(yaw0 / 2), 0, 0, Math.sin(yaw0 / 2)]);
    this.omega = new Float64Array(3);
    this.motor = new Float64Array(4).fill(params.mass * params.g / 4.0);
  }
  euler() { return quatToEuler(this.quat); }
  updateMotors(cmd, dt) {
    const a = Math.exp(-dt / this.P.motor_tau);
    for (let i = 0; i < 4; i++) this.motor[i] = a * this.motor[i] + (1.0 - a) * thrustOfCmd(Math.min(Math.max(cmd[i], 0.0), 1.0), this.P);
  }
  _deriv(vel, q, om, T, tau, rotsum, wind) {
    const P = this.P, I = this.I, R = quatToR(q);
    const d = [vel[0] - wind[0], vel[1] - wind[1], vel[2] - wind[2]];
    const va = [0, 1, 2].map(i => R[i] * d[0] + R[3 + i] * d[1] + R[6 + i] * d[2]);
    const fb = [P.kaero_xy * rotsum * va[0], P.kaero_xy * rotsum * va[1], P.kaero_z * rotsum * va[2] + T];
    const acc = [0, 1, 2].map(i => (R[3 * i] * fb[0] + R[3 * i + 1] * fb[1] + R[3 * i + 2] * fb[2]) / P.mass);
    acc[2] -= P.g;
    const Iw = [I[0] * om[0], I[1] * om[1], I[2] * om[2]];
    const cr = [om[1] * Iw[2] - om[2] * Iw[1], om[2] * Iw[0] - om[0] * Iw[2], om[0] * Iw[1] - om[1] * Iw[0]];
    const omd = [0, 1, 2].map(i => (tau[i] - cr[i]) / I[i]);
    const [w, x, y, z] = q, [p, qq, r] = om;
    const qd = [0.5 * (-x * p - y * qq - z * r), 0.5 * (w * p + y * r - z * qq), 0.5 * (w * qq + z * p - x * r), 0.5 * (w * r + x * qq - y * p)];
    return [vel, acc, qd, omd];
  }
  step(cmd, wind, dt) {
    const P = this.P;
    this.updateMotors(cmd, dt);
    const [T, tau] = motorWrench(this.motor, P);
    let rotsum = 0;
    for (let i = 0; i < 4; i++) rotsum += P.rotor_b1 * cmdOfThrust(this.motor[i], P) * P.cmd_max + P.rotor_b0;
    const s0 = [this.pos, this.vel, this.quat, this.omega];
    const f = s => this._deriv(s[1], s[2], s[3], T, tau, rotsum, wind);
    const add = (s, k, h) => s.map((a, j) => Array.from(a, (v, i) => v + h * k[j][i]));
    const k1 = f(s0), k2 = f(add(s0, k1, dt / 2)), k3 = f(add(s0, k2, dt / 2)), k4 = f(add(s0, k3, dt));
    const out = s0.map((a, j) => Float64Array.from(a, (v, i) => v + dt / 6.0 * (k1[j][i] + 2 * k2[j][i] + 2 * k3[j][i] + k4[j][i])));
    [this.pos, this.vel, this.quat, this.omega] = out;
    const n = Math.sqrt(this.quat[0] * this.quat[0] + this.quat[1] * this.quat[1] + this.quat[2] * this.quat[2] + this.quat[3] * this.quat[3]);
    for (let i = 0; i < 4; i++) this.quat[i] /= n;
    if (this.pos[2] < 0.0) {
      this.pos[2] = 0.0;
      this.vel[2] = Math.max(this.vel[2], 0.0);
      this.vel[0] = 0.0; this.vel[1] = 0.0;
    }
  }
}

export class Controller {
  constructor(gains = null, params = P0) {
    this.P = params; this.G = { ...G0, ...(gains || {}) };
    this.I = [params.Ixx, params.Iyy, params.Izz];
    this.iVz = 0; this.iVxy = [0, 0]; this.iRate = [0, 0, 0]; this.omPrev = null;
  }
  // sp = {vel_xy: [vx, vy] m/s world, alt: m, yaw: rad}; returns motor commands Float64Array(4) in [0,1]
  update(quad, sp, dt) {
    const G = this.G, P = this.P, m = P.mass, g = P.g, clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
    const [roll, pitch, yaw] = quad.euler();
    const vzSp = clip(G.kp_alt * (sp.alt - quad.pos[2]), -G.vz_max, G.vz_max);
    const eVz = vzSp - quad.vel[2];
    this.iVz = clip(this.iVz + eVz * dt, -G.i_vz_max, G.i_vz_max);
    const az = G.kp_vz * eVz + G.ki_vz * this.iVz;
    const eV = [sp.vel_xy[0] - quad.vel[0], sp.vel_xy[1] - quad.vel[1]];
    this.iVxy = this.iVxy.map((v, i) => clip(v + eV[i] * dt, -G.i_vxy_max, G.i_vxy_max));
    const axy = [0, 1].map(i => G.kp_vxy * eV[i] + G.ki_vxy * this.iVxy[i]);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const axB = cy * axy[0] + sy * axy[1], ayB = -sy * axy[0] + cy * axy[1];
    const gz = Math.max(g + az, 0.2 * g);
    const pitchSp = clip(Math.atan2(axB, gz), -G.tilt_max, G.tilt_max);
    const rollSp = clip(Math.atan2(-ayB * Math.cos(pitchSp), gz), -G.tilt_max, G.tilt_max);
    const T = m * gz / Math.max(Math.cos(roll) * Math.cos(pitch), 0.5);
    const rateSp = [clip(G.kp_att * (rollSp - roll), -G.rate_max, G.rate_max),
                    clip(G.kp_att * (pitchSp - pitch), -G.rate_max, G.rate_max),
                    clip(G.kp_yaw * wrap(sp.yaw - yaw), -G.yaw_rate_max, G.yaw_rate_max)];
    const om = quad.omega;
    if (this.omPrev === null) this.omPrev = Array.from(om);
    const imax = [G.i_rate_max, G.i_rate_max, G.i_rate_yaw_max];
    const kp = [G.kp_rate, G.kp_rate, G.kp_rate_yaw], ki = [G.ki_rate, G.ki_rate, G.ki_rate_yaw], kd = [G.kd_rate, G.kd_rate, G.kd_rate_yaw];
    const tau = [0, 1, 2].map(i => {
      const e = rateSp[i] - om[i];
      this.iRate[i] = clip(this.iRate[i] + e * dt, -imax[i], imax[i]);
      return this.I[i] * (kp[i] * e + ki[i] * this.iRate[i] - kd[i] * (om[i] - this.omPrev[i]) / dt);
    });
    this.omPrev = Array.from(om);
    const r = P.arm_length / Math.sqrt(2.0);
    const tp = 0.25 * T, rp = 0.25 / r * tau[0], pp = 0.25 / r * tau[1], yp = 0.25 * tau[2] / P.thrust2torque;
    const f = [tp - rp - pp - yp, tp - rp + pp + yp, tp + rp + pp - yp, tp + rp - pp + yp];
    const red = Math.max(Math.max(...f) - P.f_max, 0.0);
    return Float64Array.from(f, v => cmdOfThrust(clip(v - red, 0.0, P.f_max), P));
  }
}
