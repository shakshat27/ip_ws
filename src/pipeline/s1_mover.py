#!/usr/bin/env python3
"""Script 1 - MOVER (interim). Moves the arm to a named capture pose or to joint angles.

INTERIM: all motion belongs to the motion planner (script 5). Until it exists, this moves by
smooth (minimum-jerk) joint interpolation with no collision checking. Its request and answer
are the ones script 5 will keep, so swapping it later changes nothing else:
  IN : a pose name from cell.yaml (named_poses, e.g. home, or capture_poses), or six joint angles (rad)
  OUT: arrived yes/no, final joint angles, duration  -> <run>/1_move_<name>.json

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/pipeline/s1_mover.py --pose pick_overview [--run latest]
  python3 ~/ip_ws/src/pipeline/s1_mover.py --joints 0.785 -1.571 1.571 -1.571 -1.571 0
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
import cell, run_io
from ros_io import RobotIO

NAME = "s1 mover"


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5      # zero speed and acceleration at both ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", help="pose name in cell.yaml: named_poses (home) or capture_poses")
    ap.add_argument("--joints", type=float, nargs=6)
    ap.add_argument("--max-speed", type=float, default=0.5, help="rad/s, peak joint speed")
    ap.add_argument("--run", help="run folder to record into ('latest' or a name)")
    a = ap.parse_args()
    cfg = cell.load()
    if a.pose:
        group = "named_poses" if a.pose in (cfg.get("named_poses") or {}) else "capture_poses"
        target = cell.require(cfg, group, a.pose, "joints"); label = a.pose
    elif a.joints:
        target = a.joints; label = "joints"
    else:
        ap.error("give --pose or --joints")
    target = np.array(target, float)
    run_io.show(NAME, "IN", [f"target {label}: {np.round(target, 4).tolist()} rad"])

    rclpy.init(); io = RobotIO("s1_mover", command=True)
    if not io.wait_for(lambda: io.q is not None, 10):
        raise SystemExit("no /joint_states: is Isaac playing on domain 33?")
    start = io.q.copy()
    dur = max(1.5, float(np.max(np.abs(target - start))) / a.max_speed * 1.875)   # min-jerk peak = 1.875 x mean speed
    msg = JointState(); msg.name = cell.ARM_JOINTS; t0 = time.time()
    while True:
        s = min(1.0, (time.time() - t0) / dur)
        msg.position = list(start + min_jerk(s) * (target - start)); io.pub.publish(msg)
        rclpy.spin_once(io, timeout_sec=0.0); time.sleep(1 / 60)
        if s >= 1.0:
            break
    still = io.wait_still(cfg["capture"]["max_joint_speed"])
    final = io.q.copy(); err = float(np.max(np.abs(final - target)))
    arrived = bool(still and err < 0.002)
    rclpy.shutdown()

    out = dict(target=label, joints_target=target, joints_final=final, max_error_rad=err, arrived=arrived,
               duration_s=round(time.time() - t0, 2), mover="interim joint interpolation (no collision checking)")
    lines = [f"arrived {arrived}  (max joint error {np.degrees(err):.3f} deg, {out['duration_s']} s)",
             f"final joints {np.round(final, 4).tolist()} rad"]
    if a.run:
        run = run_io.resolve(a.run); path = os.path.join(run, f"1_move_{label}.json")
        run_io.write_json(path, out); lines.append(f"-> {path}")
    run_io.show(NAME, "OUT", lines)
    sys.exit(0 if arrived else 1)


if __name__ == "__main__":
    main()
