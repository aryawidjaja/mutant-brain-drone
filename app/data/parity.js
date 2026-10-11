window.PARITY = {
 "pass": true,
 "quad": {
  "tolerance_m": 0.01,
  "max_pos_err_m": 2.5219419200973697e-15,
  "pass": true,
  "provenance": {
   "git": "5558aec",
   "sim_version": "0.1.0",
   "created": "2026-10-11T18:57:29",
   "python": "3.12.12",
   "numpy": "2.5.4",
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
  "git": "7f27826-dirty",
  "created": "2026-10-11T19:21:17"
 },
 "_source": [
  "results/parity.json",
  "results/parity_brain.json"
 ]
};
