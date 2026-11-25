#!/usr/bin/env python3
"""Script 5 - PLANNER ADAPTER. One interface for every motion planner; the backend is a setting.

Plans a move from where the arm is (or --start) to a goal, with the planner chosen in cell.yaml planner.backend or
--planner. Planning and execution are separate: without --execute nothing moves. Retiming, metrics and execution
are shared code (lib/planning.py), identical for every backend, so the benchmark compares planners only.
  IN : goal = a named pose (cell.yaml named_poses / capture_poses), 6 joint angles, or a tool0 pose in the robot base
       frame (x y z m, roll pitch yaw rad); motion free | linear; start = live /joint_states or --start
  OUT: plan result (success / failure category, planning time, path, trajectory) + metrics
       -> <run>/5_plan_<label>.json; with --execute also the arrival check

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/pipeline/s5_planner.py --pose home [--planner interim|robodk] [--execute] [--run latest]
  python3 ~/ip_ws/src/pipeline/s5_planner.py --tool0 0.2 -0.54 0.10 3.1416 0 1.5708 --motion linear --planner robodk
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell, run_io, kinematics as kin
from planning import PlanRequest, get_planner, metrics, execute

NAME = "s5 planner"


def live_joints(timeout=10):
    import rclpy
    from ros_io import RobotIO
    rclpy.init(); io = RobotIO("s5_start")
    try:
        if not io.wait_for(lambda: io.q is not None, timeout):
            raise SystemExit("no /joint_states: is Isaac playing on domain 33? (or give --start)")
        return io.q.copy()
    finally:
        rclpy.shutdown()


def plain_numbers(argv):
    """argparse reads a negative number in scientific notation (e.g. -4e-06) as an option flag and stops; rewrite such
    tokens as plain decimals first (found 2026-10-23 when a tool0 pitch of -4e-06 aborted the command)."""
    out = []
    for tok in argv:
        try:
            out.append(f"{float(tok):.9f}" if tok.startswith("-") and "e" in tok.lower() else tok)
        except ValueError:
            out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pose", help="named pose in cell.yaml (named_poses or capture_poses)")
    g.add_argument("--joints", type=float, nargs=6, help="goal joint angles, rad")
    g.add_argument("--tool0", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"), help="goal tool0 pose, base frame, m / rad")
    ap.add_argument("--motion", default="free", choices=["free", "linear"])
    ap.add_argument("--planner", default=None, help="backend; default cell.yaml planner.backend")
    ap.add_argument("--start", type=float, nargs=6, default=None, help="start joints (rad); default: live /joint_states")
    ap.add_argument("--execute", action="store_true", help="send the trajectory to the arm (otherwise plan only)")
    ap.add_argument("--min-duration", type=float, default=None, help="slowest allowed: trajectory lasts at least this long (s); default cell.yaml planner.limits")
    ap.add_argument("--no-daemon", action="store_true", help="build the planner in this process instead of using the planner daemon")
    ap.add_argument("--run", default=None, help="run folder ('latest' or a name); default: a new one")
    ap.add_argument("--label", default=None, help="names the output 5_plan_<label>.json; default: the pose name, 'joints' or 'tool0'")
    a = ap.parse_args(plain_numbers(sys.argv[1:]))
    cfg = cell.load(); pc = cfg.get("planner") or {}
    backend = a.planner or pc.get("backend", "interim")

    if a.pose:
        group = "named_poses" if a.pose in (cfg.get("named_poses") or {}) else "capture_poses"
        goal = {"joints": cell.require(cfg, group, a.pose, "joints")}; label = a.pose
    elif a.joints:
        goal = {"joints": a.joints}; label = "joints"
    else:
        goal = {"pose": kin.make_T(a.tool0[:3], a.tool0[3:]).tolist()}; label = "tool0"
    label = a.label or label
    start =np.array(a.start, float) if a.start else live_joints()
    limits = dict(pc.get("limits", {}))
    if a.min_duration is not None:
        limits["min_duration_s"] = a.min_duration
    req = PlanRequest(start=start.tolist(), goal=goal, motion=a.motion, limits=limits, timeout_s=float(pc.get("timeout_s", 5.0)))
    run_io.show(NAME, "IN", [f"planner {backend}; motion {a.motion}; goal {label}: {goal}",
                             f"start {np.round(start, 4).tolist()} rad ({'given' if a.start else 'live /joint_states'})",
                             f"{'PLAN + EXECUTE' if a.execute else 'plan only (nothing moves)'}"])

    t0 = time.perf_counter(); planner = get_planner(backend, cfg, remote=not a.no_daemon); connect_s = time.perf_counter() - t0
    res = planner.plan(req)
    out = dict(planner=backend, connect_s=round(connect_s, 3), request=req.__dict__, success=res.success, failure=res.failure,
               planning_time_s=None if res.planning_time_s is None else round(res.planning_time_s, 4), notes=res.notes,
               path=res.path, trajectory=res.trajectory)
    lines = [f"success {res.success}" + (f"  failure {res.failure}: {res.notes.get('reason', '')}" if not res.success else ""),
             f"planning time {res.planning_time_s * 1000:.1f} ms (backend connect {connect_s * 1000:.0f} ms, not counted)"]
    if res.success:
        out["metrics"] = m = metrics(res.trajectory)
        lines.append(f"trajectory {m['duration_s']} s, {len(res.trajectory['t'])} points; joint path {m['joint_path_rad']} rad, "
                     f"tool path {m['tool_path_m']} m; peak {m.get('peak_speed_rad_s')} rad/s, {m.get('peak_accel_rad_s2')} rad/s2")
        if a.execute:
            out["execution"] = ex = execute(res.trajectory, cfg)
            lines.append(f"executed: arrived {ex['arrived']} (max joint error {np.degrees(ex['max_error_rad']):.3f} deg, {ex['executed_s']} s)")
            if ex.get("stopped"):
                lines.append(ex["stopped"])
    run = run_io.resolve(a.run) if a.run else run_io.new_run("plan")
    path = os.path.join(run, f"5_plan_{label}.json"); run_io.write_json(path, out); lines.append(f"-> {path}")
    run_io.show(NAME, "OUT", lines)
    sys.exit(0 if res.success and (not a.execute or out["execution"]["arrived"]) else 1)


if __name__ == "__main__":
    main()
