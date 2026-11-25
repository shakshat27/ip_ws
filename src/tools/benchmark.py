#!/usr/bin/env python3
"""Planner benchmark: the same planning queries, N times, for each backend behind the one interface (lib/planning.py).

Fair by construction (repetitions cycle through the queries, so no call repeats the one before): every backend gets identical PlanRequests (start joints, goal, motion, limits, timeout), and the
retiming and the metrics are the shared code, so only the planner differs. Plan-only (nothing moves, Isaac not needed).

Query set: frozen in evals/bench_queries.json (built once with --make-queries from successful pick runs: the recorded
requests of the pick moves, i.e. real start/goal pairs from vision-guided picks, plus home -> pick view). Reuse it for
every planner, now and later; rebuilding it changes the benchmark.
  per trial : success, failure category, planning time, and (on success) trajectory duration, joint path, tool path,
              peak joint speed / acceleration / jerk
  summary   : per planner and query kind: success rate, mean +- SD of each metric (SD over the N repetitions and the
              queries of that kind)
  OUT: evals/bench_planners_<stamp>/trials.csv, summary.json, summary.md

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/benchmark.py --make-queries                 # once
  python3 ~/ip_ws/src/tools/benchmark.py --planners robodk interim --reps 30
"""
import argparse, csv, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell, run_io
from planning import PlanRequest, get_planner, metrics

QFILE = os.path.join(cell.WS, "evals", "bench_queries.json")
KINDS = {"pre_grasp": "approach (free)", "grasp": "descent (linear)", "lift": "lift (linear)",
         "to_capture_end": "return (free)", "home_to_view": "home -> view (free)"}
METRICS = ["planning_time_s", "duration_s", "joint_path_rad", "tool_path_m", "peak_speed_rad_s", "peak_accel_rad_s2", "peak_jerk_rad_s3"]


def make_queries(cfg, since=""):
    qs = []
    for run in sorted(glob.glob(os.path.join(cell.RUNS, "*_pick"))):
        if os.path.basename(run) < since:          # runs are named <YYYYMMDD_HHMMSS>_pick: skip those made for an older robot position
            continue
        try:
            task = run_io.read_json(os.path.join(run, "0_task.json"))
        except FileNotFoundError:
            continue
        if not str(task.get("outcome", "")).startswith("picked"):
            continue
        for kind in ("pre_grasp", "grasp", "lift", "to_capture_end"):
            p = os.path.join(run, f"5_plan_{kind}.json")
            if os.path.exists(p):
                r = run_io.read_json(p)["request"]
                qs.append(dict(id=f"{os.path.basename(run)}/{kind}", kind=kind, start=r["start"], goal=r["goal"], motion=r["motion"]))
    home, view = cfg["named_poses"]["home"]["joints"], cfg["capture_poses"]["pick_overview"]["joints"]
    qs.append(dict(id="home_to_view", kind="home_to_view", start=home, goal={"joints": view}, motion="free"))
    pc = cfg.get("planner") or {}
    out = dict(made=time.strftime("%Y-%m-%d %H:%M:%S"), limits=pc.get("limits", {}), timeout_s=float(pc.get("timeout_s", 5.0)),
               source="recorded requests of successful pick runs (runs/*_pick, outcome 'picked*') + home -> pick_overview", queries=qs)
    os.makedirs(os.path.dirname(QFILE), exist_ok=True)
    run_io.write_json(QFILE, out)
    return out


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-queries", action="store_true", help="(re)build the frozen query set from the pick runs")
    ap.add_argument("--since", default="", help="with --make-queries: only pick runs whose name sorts >= this (e.g. 20261124_161400)")
    ap.add_argument("--planners", nargs="+", default=["robodk"])
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    cfg = cell.load()
    if a.make_queries:
        Q = make_queries(cfg, a.since)
        kinds = {}
        for q in Q["queries"]:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        run_io.show("benchmark", "OUT", [f"{len(Q['queries'])} queries {kinds} -> {QFILE}"])
        return
    Q = run_io.read_json(QFILE)
    out = os.path.join(cell.WS, "evals", time.strftime("bench_planners_%Y%m%d_%H%M%S")); os.makedirs(out)
    run_io.show("benchmark", "IN", [f"{len(Q['queries'])} queries from {os.path.relpath(QFILE, cell.WS)} (made {Q['made']})",
                                    f"planners {a.planners}, {a.reps} repetitions each, limits {Q['limits']}, timeout {Q['timeout_s']} s"])
    rows, setup = [], {}
    for name in a.planners:
        t0 = time.perf_counter()
        try:
            pl = get_planner(name, cfg)
        except Exception as e:
            setup[name] = dict(error=f"{type(e).__name__}: {e}"); print(f"[benchmark] {name}: not available: {e}", flush=True); continue
        setup[name] = dict(connect_s=round(time.perf_counter() - t0, 3))
        # repetitions are the OUTER loop: consecutive calls always differ. A planner that keeps state from its last call
        # (cuRobo answers an identical repeat ~15x faster than a new goal) would otherwise be flattered; in a real pick every goal is new.
        for rep in range(a.reps):
            for q in Q["queries"]:
                req = PlanRequest(start=q["start"], goal=q["goal"], motion=q["motion"], limits=Q["limits"], timeout_s=Q["timeout_s"])
                res = pl.plan(req)
                row = dict(planner=name, query=q["id"], kind=q["kind"], rep=rep, success=res.success, failure=res.failure or "",
                           planning_time_s=round(res.planning_time_s, 5) if res.planning_time_s is not None else None)
                if res.success:
                    m = metrics(res.trajectory, cfg["robot"]["dh"])
                    row.update({k: m.get(k) for k in METRICS if k != "planning_time_s"})
                rows.append(row)
            mine = [r for r in rows if r["planner"] == name and r["rep"] == rep]
            print(f"[benchmark] {name:8s} repetition {rep + 1}/{a.reps}: success {sum(r['success'] for r in mine)}/{len(mine)}, "
                  f"mean plan {np.mean([r['planning_time_s'] for r in mine]) * 1000:7.1f} ms", flush=True)
    with open(os.path.join(out, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["planner", "query", "kind", "rep", "success", "failure"] + METRICS); w.writeheader(); w.writerows(rows)
    summary = dict(queries=os.path.relpath(QFILE, cell.WS), reps=a.reps, limits=Q["limits"], setup=setup, results={})
    lines = ["| planner | query kind | n | success | planning time ms | duration s | joint path rad | tool path m | peak accel rad/s² |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in a.planners:
        for kind in KINDS:
            R = [r for r in rows if r["planner"] == name and r["kind"] == kind]
            if not R:
                continue
            ok = [r for r in R if r["success"]]
            s = dict(n=len(R), success_rate=round(len(ok) / len(R), 4),
                     failures={f: sum(r["failure"] == f for r in R) for f in {r["failure"] for r in R if r["failure"]}})
            for k in METRICS:
                mu, sd = ms([r.get(k) for r in (R if k == "planning_time_s" else ok)])
                s[k] = None if mu is None else dict(mean=round(mu, 6), sd=round(sd, 6))
            summary["results"].setdefault(name, {})[kind] = s
            f = lambda k, sc=1.0, d=1: "–" if s[k] is None else f"{s[k]['mean'] * sc:.{d}f} ± {s[k]['sd'] * sc:.{d}f}"
            lines.append(f"| {name} | {KINDS[kind]} | {len(R)} | {s['success_rate'] * 100:.0f} % | {f('planning_time_s', 1000, 1)} | "
                         f"{f('duration_s', 1, 2)} | {f('joint_path_rad', 1, 3)} | {f('tool_path_m', 1, 3)} | {f('peak_accel_rad_s2', 1, 2)} |")
    run_io.write_json(os.path.join(out, "summary.json"), summary)
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    run_io.show("benchmark", "OUT", lines + [f"-> {out}/ (trials.csv, summary.json, summary.md)"])


if __name__ == "__main__":
    main()
