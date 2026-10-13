window.PARITY = {
 "pass": true,
 "quad": {
  "tolerance_m": 0.01,
  "max_pos_err_m": 2.5219419200973697e-15,
  "pass": true,
  "provenance": {
   "git": "9c7b912",
   "sim_version": "0.1.0",
   "created": "2026-10-13T16:41:37",
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
  "tolerances": "Module outputs are compared on recorded inputs (exact or <= 1e-9); full closed-loop missions are compared on the\ntrajectory with a stated tolerance (closed loops amplify floating-point order differences between numpy and V8):\n  scenarios, hashes, worlds ............ exact\n  eye panorama pixels .................. <= 0.5 % of pixels differ, by <= 1 level\n  object looks (grey) .................. max |diff| <= 1e-9\n  MB familiarity on Python's looks ..... exact;  on each side's own looks + same noise: max |diff| <= 0.02\n  path integrator (recorded inputs) .... heading <= 1e-9 rad, home vector <= 1e-6 m\n  six arbiters (recorded obs) .......... heading <= 1e-9 rad, commit <= 1e-9\n  avoidance reflex ..................... <= 1e-12;  mind-reader strings: exact\n  missions (Python batch vs JS batch 1)  position <= 0.01 m for the first 20 s; same success/stop flags\n    v1: all six arbiters closed loop; v2 (brain.version 2, all six modules, RingGated and visibility-gated rivals,\n    HeadingRing and KalmanHeading): same rule, run to 90 s so the take-off learning flight, the pad lock and the\n    touchdown are inside the compared window; landing flags (landed_on_pad) must agree too; post-study adaptive_search\n    (Study 2 modules + adaptive_search, frozen v2 ring): a tent and a backpack mission, same rule, size estimate and\n    search altitude equal to 1e-9\n  eye RGB crops (v2 front end input) .... max |diff| <= 1e-9\nRun: uv run python scripts/parity_brain.py\n",
  "git": "a6f477b-dirty",
  "created": "2026-10-13T21:07:44"
 },
 "_source": [
  "results/parity.json",
  "results/parity_brain.json"
 ]
};
