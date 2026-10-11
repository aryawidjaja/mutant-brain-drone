"""results/parity.json + results/parity_brain.json -> app/data/parity.js (window.PARITY) for the app's verification banner."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    q = json.loads((ROOT / "results/parity.json").read_text())
    b = json.loads((ROOT / "results/parity_brain.json").read_text())
    pick = lambda d: {k: d[k] for k in d if k in ("pass", "max_pos_err_m", "tolerance_m", "tolerances", "provenance", "git", "created")}
    return {"pass": bool(q["pass"] and b["pass"]), "quad": pick(q), "brain": pick(b),
            "_source": ["results/parity.json", "results/parity_brain.json"]}


if __name__ == "__main__":
    p = build()
    (ROOT / "app/data/parity.js").write_text("window.PARITY = " + json.dumps(p, indent=1) + ";\n")
    assert p["pass"] is (json.loads((ROOT / "results/parity.json").read_text())["pass"] and json.loads((ROOT / "results/parity_brain.json").read_text())["pass"])
    print("parity.js written, pass =", p["pass"])
