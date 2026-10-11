"""Provenance stamp for every result file: git hash, code hashes, simulator version, time, library versions."""
import datetime, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np

SIM_VERSION = "0.1.0"  # bump when quad/brain/mission equations change
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def git_hash():
    """Short HEAD hash, "-dirty" if scripts/ or app/sim/ differ from HEAD in any way (modified, staged or untracked)."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "scripts", "app/sim"], cwd=ROOT, text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def code_sha256():
    """sha256 of every loaded module that lives under scripts/ (the code this run actually imported)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py") and Path(f).resolve().is_relative_to(SCRIPTS):
            p = Path(f).resolve()
            out[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def stamp(**extra):
    return {"git": git_hash(), "sim_version": SIM_VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, "code_sha256": code_sha256(), **extra}


def assert_reproduces(path, new, skip=("provenance",)):
    """Assert that every key of the existing result file at `path` (except `skip`) equals `new` exactly (after a JSON
    round trip). Guards pre-registered numbers when a script is re-run to add post hoc keys."""
    path = Path(path)
    if not path.exists():
        return
    old = json.loads(path.read_text())
    new = json.loads(json.dumps(new))
    bad = [k for k in old if k not in skip and old[k] != new.get(k)]
    assert not bad, f"{path.name}: pre-registered keys changed on re-run: {bad}"


if __name__ == "__main__":
    s = stamp(seed=0)
    assert s["git"] and s["sim_version"] and s["seed"] == 0
    assert "scripts/provenance.py" in s["code_sha256"] and all(len(v) == 64 for v in s["code_sha256"].values())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        p.write_text(json.dumps({"a": 0.1, "b": [1, 2], "provenance": 1}))
        assert_reproduces(p, {"a": 0.1, "b": [1, 2], "provenance": 2, "posthoc": 3})
        try:
            assert_reproduces(p, {"a": 0.1 + 1e-15, "b": [1, 2]})
            raise RuntimeError("change not caught")
        except AssertionError:
            pass
    print(s)
