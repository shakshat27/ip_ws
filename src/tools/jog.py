#!/usr/bin/env python3
"""Jog the arm in small steps, like a teach pendant: one joint, or the TCP along / about an axis of the base or tool frame.

The goal is computed from the arm's current joints (/joint_states) with OUR UR10 kinematics; a Cartesian step is solved
by Newton steps from the current joints (local IK: the arm stays in its configuration), so it needs no planner and no
RoboDK. The move goes through the shared executor (lib/planning.py: same retiming and contact stop as every other move),
slowly. Refused: a step that would flip the configuration (a joint changing > 0.5 rad for one step) or leave the joint limits.
  IN : /joint_states; cell.yaml robot.dh, gripper.tool0_T_tcp (the point a Cartesian step moves / rotates about)
  OUT: the arm moved; one line with the new joints and TCP

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/jog.py --joint 3 --step -5          # joint 3 (elbow) by -5 deg
  python3 ~/ip_ws/src/tools/jog.py --axis z --step 10           # TCP 10 mm up, base frame
  python3 ~/ip_ws/src/tools/jog.py --axis rz --step 5 --frame tool
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import cell, kinematics as kin
from planning import retime, execute

AXES = ["x", "y", "z", "rx", "ry", "rz"]
LIMIT = 2 * math.pi                               # UR10: every joint +-360 deg
SPEED = {"max_joint_speed": 0.25, "min_duration_s": 0.4}   # jogging is slow on purpose


def live_q():
    import rclpy
    from ros_io import RobotIO
    rclpy.init(); io = RobotIO("jog_read")
    try:
        if not io.wait_for(lambda: io.q is not None, 8):
            raise SystemExit("no /joint_states: is Isaac playing on domain 33?")
        return io.q.copy()
    finally:
        rclpy.shutdown()


def pose_error(T, G):
    dp = G[:3, 3] - T[:3, 3]
    dr, _ = cv2.Rodrigues(G[:3, :3] @ T[:3, :3].T)
    return np.r_[dp, dr.ravel()]


def local_ik(q, G, dh, iters=30):
    """Newton steps (numeric Jacobian) from q to the flange pose G; None if it does not converge."""
    q = np.array(q, float)
    for _ in range(iters):
        e = pose_error(kin.fk(q, dh), G)
        if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
            return q
        J = np.zeros((6, 6)); h = 1e-6
        for i in range(6):
            dq = q.copy(); dq[i] += h
            J[:, i] = (pose_error(kin.fk(q, dh), G) - pose_error(kin.fk(dq, dh), G)) / h
        q = q + np.linalg.lstsq(J, e, rcond=None)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--joint", type=int, choices=range(1, 7), help="joint number 1..6 (base .. wrist 3)")
    g.add_argument("--axis", choices=AXES, help="TCP axis: translate along x/y/z or rotate about rx/ry/rz")
    ap.add_argument("--step", type=float, required=True, help="deg for a joint or rotation, mm for a translation (signed)")
    ap.add_argument("--frame", default="base", choices=["base", "tool"], help="Cartesian frame (rotations are about the TCP)")
    a = ap.parse_args()
    cfg = cell.load(); dh = cfg["robot"]["dh"]
    q0 = live_q(); goal = q0.copy()
    if a.joint:
        if abs(a.step) > 30:
            raise SystemExit("joint step limited to 30 deg")
        goal[a.joint - 1] += math.radians(a.step); what = f"J{a.joint} {a.step:+g} deg"
    else:
        if a.axis in ("x", "y", "z") and abs(a.step) > 100 or a.axis.startswith("r") and abs(a.step) > 30:
            raise SystemExit("step limited to 100 mm / 30 deg")
        tcp = kin.T_from_cfg(cfg["gripper"]["tool0_T_tcp"])
        T = kin.fk(q0, dh) @ tcp                                  # current TCP in the base frame
        k = "xyz".index(a.axis[-1]); unit = np.eye(3)[k]
        Rf = np.eye(3) if a.frame == "base" else T[:3, :3]        # axis expressed in base or tool frame
        G = T.copy()
        if len(a.axis) == 1:
            G[:3, 3] += Rf @ unit * a.step / 1000.0
        else:
            G[:3, :3] = cv2.Rodrigues(Rf @ unit * math.radians(a.step))[0] @ T[:3, :3]
        q = local_ik(q0, G @ np.linalg.inv(tcp), dh)
        if q is None:
            raise SystemExit(f"no solution for {a.axis} {a.step:+g} ({a.frame}): near a singularity or out of reach")
        goal = q; what = f"TCP {a.axis} {a.step:+g} {'mm' if len(a.axis) == 1 else 'deg'} ({a.frame})"
    if np.max(np.abs(goal - q0)) > 0.5:
        raise SystemExit(f"refused: {what} needs a joint to turn {math.degrees(np.max(np.abs(goal - q0))):.0f} deg (configuration change)")
    if np.any(np.abs(goal) > LIMIT):
        raise SystemExit(f"refused: {what} leaves the joint limits")
    ex = execute(retime([q0, goal], SPEED), cfg, node_name="jog")
    T = kin.fk(ex["final_joints"], dh) @ kin.T_from_cfg(cfg["gripper"]["tool0_T_tcp"])
    rpy = np.degrees(kin.R_to_rpy(T[:3, :3]))
    status = "ok" if ex["arrived"] else ("STOPPED: " + ex["stopped"] if ex.get("stopped") else "did not arrive")
    print(f"jog {what}: {status} | joints deg {np.round(np.degrees(ex['final_joints']), 1).tolist()} | "
          f"TCP mm {np.round(T[:3, 3] * 1000, 1).tolist()} rpy deg {np.round(rpy, 1).tolist()}", flush=True)
    sys.exit(0 if ex["arrived"] else 1)


if __name__ == "__main__":
    main()
