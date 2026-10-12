"""Run sharded experiment stages on Kaggle CPU kernels (private), from the committed tree.

  kaggle_launch.py code                         package `git archive HEAD` as the private dataset <user>/mbd-code
  kaggle_launch.py push  <runner> <stage> N     push N kernels: python scripts/<runner> run <stage> --shard i --nshards N
  kaggle_launch.py status <stage> N             one line per kernel
  kaggle_launch.py pull  <runner> <stage> N     download finished outputs into results/<out>/<stage>/ (skips existing)
Credentials: the Kaggle CLI's own configuration (used in place; never read or copied here).
"""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "maryawijaya"
CODE = f"{USER}/mbd-code"
OUTDIR = {"run_3d.py": "results/3d", "run_v2.py": "results/v2", "kt7_stubbornness_v2.py": "results/kaggle_kt7"}
PIP = "numpy==2.5.4 scipy==1.18.1 pandas==3.0.6 pyarrow==26.0.0"

RUN_PY = '''import os, subprocess, sys, tarfile, glob
os.makedirs("/kaggle/working/repo", exist_ok=True)
tars = glob.glob("/kaggle/input/**/code.tar.gz", recursive=True)
if tars:
    tarfile.open(tars[0]).extractall("/kaggle/working/repo")
else:   # Kaggle unpacks archives on upload: copy the unpacked tree
    top = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/scripts/provenance.py", recursive=True)[0]))
    subprocess.run(f"cp -r {{top}}/. /kaggle/working/repo/", shell=True, check=True)
subprocess.run("pip install -q {pip}", shell=True, check=True)
os.chdir("/kaggle/working/repo")
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", KT_PROCS="4")
r = subprocess.run([sys.executable] + {cmd!r}, env=env)
subprocess.run("mkdir -p /kaggle/working/out && cp -r {out}/* /kaggle/working/out/ 2>/dev/null; {extra} true", shell=True)
subprocess.run("rm -rf /kaggle/working/repo", shell=True)
sys.exit(r.returncode)
'''


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def code():
    head = sh("git rev-parse --short HEAD", cwd=ROOT).stdout.strip()
    if sh("git status --porcelain -- scripts app/sim schema", cwd=ROOT).stdout.strip():
        print("note: uncommitted changes are NOT included; packaging HEAD only")
    with tempfile.TemporaryDirectory() as d:
        sh(f"git archive --format=tar.gz -o {d}/code.tar.gz HEAD", cwd=ROOT, check=True)
        (Path(d) / "dataset-metadata.json").write_text(json.dumps({"title": "mbd-code", "id": CODE, "licenses": [{"name": "other"}]}))
        exists = sh(f"kaggle datasets status {CODE}").returncode == 0
        r = sh(f"kaggle datasets version -p {d} -m {head} -q" if exists else f"kaggle datasets create -p {d} -q")
        print(r.stdout[-300:], r.stderr[-300:])
    print("code dataset at", head)


def kid(stage, i):
    return f"{USER}/mbd-{stage.replace('_', '-')}-{i}"


def push(runner, stage, n, only=None):
    for i in (range(n) if only is None else [only]):
        with tempfile.TemporaryDirectory() as d:
            cmd = [f"scripts/{runner}", "run", stage, "--shard", str(i), "--nshards", str(n), "--procs", "4"] \
                if runner != "kt7_stubbornness_v2.py" else [f"scripts/{runner}", "--test", "--procs", "4"]
            extra = "cp -r results/*.json results/*.png /kaggle/working/out/ 2>/dev/null;" if runner == "kt7_stubbornness_v2.py" else ""
            (Path(d) / "run.py").write_text(RUN_PY.format(pip=PIP, cmd=cmd, out=OUTDIR.get(runner, "results"), extra=extra))
            (Path(d) / "kernel-metadata.json").write_text(json.dumps({
                "id": kid(stage, i), "title": kid(stage, i).split("/")[1], "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [CODE], "competition_sources": [], "kernel_sources": [], "model_sources": []}))
            r = sh(f"kaggle kernels push -p {d}")
            print(i, (r.stdout + r.stderr).strip().splitlines()[-1:] or "")


def status(stage, n):
    for i in range(n):
        print(i, sh(f"kaggle kernels status {kid(stage, i)}").stdout.strip()[-80:])


def pull(runner, stage, n):
    dest = ROOT / OUTDIR.get(runner, "results") / stage
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        with tempfile.TemporaryDirectory() as d:
            r = sh(f"kaggle kernels output {kid(stage, i)} -p {d} -q")
            got = list(Path(d).rglob("*.parquet")) + list(Path(d).rglob("*.json")) + list(Path(d).rglob("*.png"))
            for f in got:
                if not (dest / f.name).exists():
                    (dest / f.name).write_bytes(f.read_bytes())
            print(i, len(got), "files", r.stderr.strip()[-120:])


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        assert RUN_PY.format(pip=PIP, cmd=["x"], out="results", extra="").count("extractall") == 1
        assert kid("tune_r1", 3) == f"{USER}/mbd-tune-r1-3"
        print("kaggle_launch self-check OK"); sys.exit(0)
    {"code": lambda: code(), "push": lambda: push(a[1], a[2], int(a[3])), "push-one": lambda: push(a[1], a[2], int(a[3]), int(a[4])), "status": lambda: status(a[1], int(a[2])),
     "pull": lambda: pull(a[1], a[2], int(a[3]))}[a[0]]()
