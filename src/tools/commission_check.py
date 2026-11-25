#!/usr/bin/env python3
"""REFEREE TOOL. Checks the robot's own kinematics against the simulator. Its output must never be
copied into cell.yaml as a parameter: the camera offset comes from hand-eye calibration and the
surfaces from teaching. It only answers "is our maths right?" and records the referee survey.

One-time commissioning check: does the robot's own kinematics agree with the simulator?

Compares, at the current arm pose:
  FK(joint angles)                     vs  Isaac's wrist_3_link   -> base and flange conventions
  FK(joint angles) @ tool0_T_camera    vs  Isaac's camera prim    -> the camera bracket offset
and prints where the robot base stands in Isaac's world (the "cell survey").
Isaac's poses are used here ONLY as a referee, the way a laser tracker would be used to
commission a real cell. Nothing at run time reads them.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/commission_check.py          # run with Isaac playing; try 2-3 different arm poses
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib"))

import math, os, time
import numpy as np, yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import kinematics as kin

CFG = os.path.expanduser("~/ip_ws/config/cell.yaml")
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


def main():
    cfg = yaml.safe_load(open(CFG))
    rclpy.init(); n = Node("commission_check"); tf, js = {}, {}
    n.create_subscription(TFMessage, "/tf", lambda m: [tf.__setitem__(t.child_frame_id, (t.header.frame_id, t.transform)) for t in m.transforms], 50)
    n.create_subscription(JointState, "/joint_states", lambda m: js.update(q=[m.position[m.name.index(j)] for j in ARM]), 10)
    cam_frame = cfg["camera"]["frame"]
    t0 = time.time()
    while time.time() - t0 < 20 and not ("q" in js and {"base_link", "wrist_3_link", cam_frame} <= set(tf)):
        rclpy.spin_once(n, timeout_sec=0.05)
    rclpy.shutdown()
    if "q" not in js or not {"base_link", "wrist_3_link", cam_frame} <= set(tf):
        print("missing data: Isaac must be playing on domain 33 with joint states and /tf"); return

    def world(name):
        parent, tr = tf[name]
        T = np.eye(4); T[:3, :3] = kin.quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        return T if parent in ("world", "World") else world(parent) @ T

    q = js["q"]; F = kin.fk(q, cfg.get("robot", {}).get("dh", kin.UR10_DH))
    W_bl, W_w3, W_cam = world("base_link"), world("wrist_3_link"), world(cam_frame)
    print("joint angles (deg):", [round(math.degrees(a), 2) for a in q])

    # 1. base convention: UR's DH base is either Isaac's base_link, or base_link turned 180 deg about Z
    best = None
    for name, C in (("same as Isaac base_link", np.eye(4)), ("Isaac base_link turned 180 deg about Z", RZ_PI)):
        e = np.linalg.norm((W_bl @ C @ F)[:3, 3] - W_w3[:3, 3]) * 1000
        print(f"  base = {name:40s} flange position error {e:8.2f} mm")
        if best is None or e < best[0]:
            best = (e, name, C)
    e_pos, base_name, C = best
    W_base = W_bl @ C
    flange = W_base @ F
    R_off = flange[:3, :3].T @ W_w3[:3, :3]
    print(f"-> base convention: {base_name}; flange position error {e_pos:.2f} mm")
    print(f"   Isaac wrist_3_link vs UR tool0 rotation: {angle_deg(flange[:3, :3], W_w3[:3, :3]):.2f} deg, "
          f"as roll/pitch/yaw {[round(math.degrees(a), 1) for a in kin.R_to_rpy(R_off)]} deg")

    # 2. camera: nominal bracket offset vs the simulated camera
    he = cfg.get("hand_eye", {}).get("tool0_T_camera")
    if he:
        cam = flange @ kin.T_from_cfg(he)
        print(f"-> camera from kinematics vs simulator: position {np.linalg.norm(cam[:3, 3] - W_cam[:3, 3]) * 1000:.2f} mm, "
              f"rotation {angle_deg(cam[:3, :3], W_cam[:3, :3]):.2f} deg")
    exact = np.linalg.inv(flange) @ W_cam
    print(f"   camera in tool0 according to the simulator: xyz {exact[:3, 3].round(4).tolist()} m, "
          f"rpy {[round(a, 4) for a in kin.R_to_rpy(exact[:3, :3])]} rad")

    # 3. cell survey: where the robot base stands in Isaac's world (referee side only)
    print(f"-> survey world_T_base: xyz {W_base[:3, 3].round(4).tolist()} m, rpy {[round(a, 4) for a in kin.R_to_rpy(W_base[:3, :3])]} rad")


if __name__ == "__main__":
    main()
