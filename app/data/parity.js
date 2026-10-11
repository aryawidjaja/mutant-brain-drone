window.PARITY = {
 "pass": true,
 "quad": {
  "tolerance_m": 0.01,
  "max_pos_err_m": 2.5219419200973697e-15,
  "pass": true,
  "provenance": {
   "git": "4476594",
   "sim_version": "0.1.0",
   "created": "2026-10-12T00:31:25",
   "python": "3.12.12",
   "numpy": "2.5.4",
   "code_sha256": {
    "scripts/dryden.py": "cc577521940146fd791e45606f0773fc0b67a97679209dfd394cac3d75cbe4c8",
    "scripts/parity_check.py": "76e2443390b7b4de9a3ee6ca93622677e9267acd7d3bce19e080d150f446f3e7",
    "scripts/provenance.py": "fcc5a30df4c62e294473eb3307ba3d77284dcb63a657fe047b35f2bd8aea86d0",
    "scripts/quad.py": "78dafc0ca45bcb7d82c3e7d9d04b3269cbea7e8d4b82bc5e1e1a2520b22a4703",
    "scripts/rng.py": "532bb3605248c236de450ef4e1d44e984448b9aaec281db3c6faec52aab84001"
   },
   "seed": [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1001,
    1002,
    1003,
    0,
    7
   ],
   "param_source": "results/vehicle_params.json",
   "node": "v25.4.0"
  }
 },
 "brain": {
  "pass": true,
  "tolerances": "Module outputs are compared on recorded inputs (exact or <= 1e-9); full closed-loop missions are compared on the\ntrajectory with a stated tolerance (closed loops amplify floating-point order differences between numpy and V8):\n  scenarios, hashes, worlds ............ exact\n  eye panorama pixels .................. <= 0.5 % of pixels differ, by <= 1 level\n  object looks (grey) .................. max |diff| <= 1e-9\n  MB familiarity on Python's looks ..... exact;  on each side's own looks + same noise: max |diff| <= 0.02\n  path integrator (recorded inputs) .... heading <= 1e-9 rad, home vector <= 1e-6 m\n  six arbiters (recorded obs) .......... heading <= 1e-9 rad, commit <= 1e-9\n  avoidance reflex ..................... <= 1e-12;  mind-reader strings: exact\n  missions (Python batch vs JS batch 1)  position <= 0.01 m for the first 20 s; same success/stop flags\nRun: uv run python scripts/parity_brain.py\n",
  "git": "4476594",
  "created": "2026-10-12T00:32:04"
 },
 "_source": [
  "results/parity.json",
  "results/parity_brain.json"
 ]
};
