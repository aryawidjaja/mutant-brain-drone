"""Surrogate of the Shiu-model yaw circuit (the author's earlier FlyWire v783 fly project) for live use in the browser.

Input data: ../flight-test-the-fly/results/phase1_bode.json, wirings.real: lock-in transfer H(f) of
Delta = DNa02_R - DNa02_L (Hz) re yaw rate r (deg/s), 13 frequencies 0.1-20 Hz, 5 seeds, with coherence.
Model (3 parameters): H(s) = -K exp(-s tau) / (1 + s / wc). Fitted by coherence-weighted complex least squares
(scipy least_squares on real and imaginary residuals, normalised by |H|). Reported: RMS gain error (dB) and RMS phase
error (deg) over the 13 points. Discrete form used in Python and app/sim/yaw_surrogate.js (same equations):
delay line of round(tau/dt) samples, then the exact zero-order-hold lag y <- a y + (1-a)(-K u_delayed), a = exp(-wc dt).
Labelled everywhere as "surrogate of the Shiu-model yaw circuit"; units are those of the source (Hz per deg/s), only
phase/bandwidth carry meaning.
Run: uv run python scripts/fit_yaw_surrogate.py  -> results/yaw_surrogate.json
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from provenance import stamp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "flight-test-the-fly" / "results" / "phase1_bode.json"


def model(f, K, tau, wc):
    s = 2j * np.pi * np.asarray(f)
    return -K * np.exp(-s * tau) / (1 + s / wc)


class YawSurrogate:
    """Discrete surrogate: step(r_degps) -> Delta (Hz). Batch 1 or numpy arrays."""

    def __init__(self, fit, dt=0.002):
        self.K, self.n = fit["K"], int(round(fit["tau_s"] / dt))
        self.a = np.exp(-fit["wc_radps"] * dt)
        self.buf, self.y = [0.0] * self.n, 0.0

    def step(self, r):
        self.buf.append(r)
        u = self.buf.pop(0)
        self.y = self.a * self.y + (1.0 - self.a) * (-self.K * u)
        return self.y


def fit():
    d = json.loads(SRC.read_text())
    r = d["wirings"]["real"]
    f, H, coh = np.array(r["f"]), np.array(r["H_re"]) + 1j * np.array(r["H_im"]), np.array(r["coh"])
    w = np.sqrt(coh) / np.abs(H)

    def res(p):
        e = (model(f, *p) - H) * w
        return np.concatenate([e.real, e.imag])

    p = least_squares(res, [0.68, 0.02, 2 * np.pi * 15], bounds=([0, 0, 1], [5, 0.2, 2000])).x
    Hm = model(f, *p)
    db = 20 * np.log10(np.abs(Hm) / np.abs(H))
    dph = np.degrees(np.angle(Hm / H))
    return {"K": float(p[0]), "tau_s": float(p[1]), "wc_radps": float(p[2]), "fc_hz": float(p[2] / (2 * np.pi)),
            "rms_gain_err_db": float(np.sqrt(np.mean(db ** 2))), "max_abs_gain_err_db": float(np.abs(db).max()),
            "rms_phase_err_deg": float(np.sqrt(np.mean(dph ** 2))), "max_abs_phase_err_deg": float(np.abs(dph).max()),
            "per_freq": {"f_hz": f.tolist(), "gain_data": np.abs(H).tolist(), "gain_fit": np.abs(Hm).tolist(),
                         "phase_data_deg": np.degrees(np.angle(H)).tolist(), "phase_fit_deg": np.degrees(np.angle(Hm)).tolist(),
                         "coherence": coh.tolist()},
            "source_meta": {"file": str(SRC.relative_to(ROOT.parent)), "git_hash": d["meta"]["git_hash"],
                            "created": d["meta"]["created"], "data_version": d["meta"]["data_version"],
                            "conventions": d["meta"]["conventions"]}}


def sine_gain(fitd, f_hz, dt=0.002, T=None):
    """Simulated discrete surrogate gain and phase at f_hz (lock-in over whole cycles after a transient)."""
    s = YawSurrogate(fitd, dt)
    T = T or max(4.0, 20.0 / f_hz)
    t = np.arange(int(T / dt)) * dt
    y = np.array([s.step(np.sin(2 * np.pi * f_hz * tt)) for tt in t])
    keep = t >= T / 2
    z = 2 * np.mean(y[keep] * np.exp(-2j * np.pi * f_hz * t[keep]))
    return z / (-1j)  # input sin = Im(e^{jwt}) -> complex response relative to the input phasor


if __name__ == "__main__":
    fd = fit()
    fd["label"] = "surrogate of the Shiu-model yaw circuit (fitted transfer function, not the spiking model)"
    fd["model"] = "H(s) = -K exp(-s tau) / (1 + s/wc); input yaw rate (deg/s), output DNa02 R-L rate difference (Hz)"
    # discrete implementation matches the continuous fit within 0.5 dB / 10 deg up to 7.7 Hz (dt = 2 ms delay rounding)
    chk = []
    for fq in (0.5, 2.8986, 7.6923):
        z, zc = sine_gain(fd, fq), model(fq, fd["K"], fd["tau_s"], fd["wc_radps"])
        chk.append({"f_hz": fq, "db": float(20 * np.log10(abs(z) / abs(zc))), "deg": float(np.degrees(np.angle(z / zc)))})
    fd["discrete_vs_continuous"] = chk
    assert all(abs(c["db"]) < 0.5 and abs(c["deg"]) < 10 for c in chk), chk
    assert fd["rms_gain_err_db"] < 3.0 and fd["rms_phase_err_deg"] < 20.0, fd
    # JS port (app/sim/yaw_surrogate.js) gives the same output sequence
    u = np.sin(2 * np.pi * 3.0 * np.arange(2000) * 0.002).tolist()
    s = YawSurrogate(fd)
    py = [s.step(x) for x in u]
    js = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e",
         "import {YawSurrogate} from './app/sim/yaw_surrogate.js';"
         "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{const {fit,u}=JSON.parse(d);"
         "const s=new YawSurrogate(fit,0.002);process.stdout.write(JSON.stringify(u.map(x=>s.step(x))));});"],
        input=json.dumps({"fit": {k: fd[k] for k in ("K", "tau_s", "wc_radps")}, "u": u}), capture_output=True, text=True,
        cwd=ROOT, check=True).stdout)
    fd["js_parity_max_abs_err"] = float(np.abs(np.array(js) - np.array(py)).max())
    assert fd["js_parity_max_abs_err"] < 1e-12
    fd["provenance"] = stamp(seed=None, param_source=fd["source_meta"]["file"])
    (ROOT / "results" / "yaw_surrogate.json").write_text(json.dumps(fd, indent=1))
    print({k: fd[k] for k in ("K", "tau_s", "fc_hz", "rms_gain_err_db", "max_abs_gain_err_db", "rms_phase_err_deg",
                              "max_abs_phase_err_deg", "js_parity_max_abs_err")})
    print("yaw surrogate OK")
