"""Provenance stamp for every result file: git hash, simulator version, time, library versions."""
import datetime, platform, subprocess
import numpy as np

SIM_VERSION = "0.1.0"  # bump when quad/brain/mission equations change


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "scripts", "app/sim"], stderr=subprocess.DEVNULL)
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def stamp(**extra):
    return {"git": git_hash(), "sim_version": SIM_VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, **extra}


if __name__ == "__main__":
    s = stamp(seed=0)
    assert s["git"] and s["sim_version"] and s["seed"] == 0
    print(s)
