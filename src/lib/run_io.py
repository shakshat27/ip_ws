"""Run folders: every script writes what it produced into ~/ip_ws/runs/<run>/<step>/ and prints
what it received (IN) and what it sent (OUT). Any step can be rerun on a saved folder."""
import json, os, time
import numpy as np
from cell import RUNS


def new_run(tag="run"):
    path = os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S") + f"_{tag}")
    os.makedirs(path, exist_ok=True)
    return path


def resolve(run):
    """'latest' -> newest run folder; a bare name -> under ~/ip_ws/runs; a path -> itself."""
    if run in (None, "latest"):
        runs = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))) if os.path.isdir(RUNS) else []
        if not runs:
            raise SystemExit("no runs yet in ~/ip_ws/runs")
        return os.path.join(RUNS, runs[-1])
    return run if os.path.isdir(run) else os.path.join(RUNS, run)


def step_dir(run, step):
    path = os.path.join(run, step)
    os.makedirs(path, exist_ok=True)
    return path


def latest_step(run, prefix):
    cands = sorted(d for d in os.listdir(run) if d.startswith(prefix))
    if not cands:
        raise SystemExit(f"no '{prefix}*' step in {run}")
    return os.path.join(run, cands[-1])


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_plain)


def read_json(path):
    return json.load(open(path))


def show(script, direction, lines):
    """Uniform transparent log: [s3 detect] IN / OUT followed by indented lines."""
    print(f"[{script}] {direction}", flush=True)
    for line in lines:
        print(f"    {line}", flush=True)
