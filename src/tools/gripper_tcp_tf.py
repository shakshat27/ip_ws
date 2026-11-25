#!/usr/bin/env python3
"""The gripper's TCP from the ROBOT MODEL, the way a real robot knows it: finger link poses from /tf (joint angles +
kinematics; on a real cell robot_state_publisher with the URDF) plus the pad centre in each finger link (gripper CAD,
cell.yaml gripper.pad). Replaces the camera estimate (measure_gripper.py, PC-10).

For each finger_joint angle in the sweep it commands the gripper, waits, then reads wrist_3_link -> left/right_inner_finger:
  pad centres  = finger link pose x pad centre in that link
  TCP          = midpoint of the two pad centres, in tool0 (= /tf wrist_3_link, verified identical to our FK tool0)
  closing axis = direction between the pad centres (its yaw in tool0 X-Y)
  gap          = pad-centre distance - pad thickness (inner faces)
  IN : /tf, /joint_states (Isaac playing), cell.yaml gripper.pad
  OUT: ~/ip_ws/calib/gripper_tf_<stamp>.json; with --write: cell.yaml gripper.tool0_T_tcp (open), closing_axis_yaw_in_tool0,
       opening_table (backup <file>.<stamp>.bak next to it). Moves only the fingers; leaves them open.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/gripper_tcp_tf.py [--write]
"""
import argparse, math, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell, run_io

NAME = "gripper tcp (tf)"
HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def read_tf(node, buf, pad):
    """Pad centres (tool0 frame) from the current /tf, averaged over 0.5 s."""
    import rclpy
    from scipy.spatial.transform import Rotation as Rot
    acc = {"left": [], "right": []}
    end = time.time() + 0.5
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
        for s in acc:
            t = buf.lookup_transform("wrist_3_link", f"{s}_inner_finger", rclpy.time.Time()).transform
            M = np.eye(4); M[:3, :3] = Rot.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
            M[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
            acc[s].append((M @ np.append(pad[f"centre_in_{s}_inner_finger"], 1))[:3])
    return {s: np.mean(v, 0) for s, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store the result in cell.yaml (backup kept)")
    a = ap.parse_args()
    cfg = cell.load(); pad = cell.require(cfg, "gripper", "pad")
    import rclpy
    from ros_io import RobotIO
    from tf2_ros import Buffer, TransformListener
    run_io.show(NAME, "IN", ["/tf wrist_3_link -> left/right_inner_finger; pad centres from cell.yaml gripper.pad (gripper CAD)",
                             f"finger_joint sweep {SWEEP} rad (fingers only)"])
    rclpy.init(); node = RobotIO("gripper_tcp_tf"); buf = Buffer(); TransformListener(buf, node)
    try:
        if not node.wait_for(lambda: buf.can_transform("wrist_3_link", "left_inner_finger", rclpy.time.Time()), 10):
            raise SystemExit("no /tf for the finger links: is Isaac playing on domain 33?")
        rows = []
        for q in SWEEP:
            r = subprocess.run([sys.executable, os.path.join(HERE, "gripper.py"), "--pos", str(q), "--hold", "1.2"], capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"gripper command failed: {r.stderr[-300:]}")
            c = read_tf(node, buf, pad)
            mid = (c["left"] + c["right"]) / 2; d = c["right"] - c["left"]
            yaw = (math.atan2(d[1], d[0]) + math.pi / 2) % math.pi - math.pi / 2      # an axis has no sign: -90..90 deg
            rows.append(dict(finger_joint_rad=q, tcp_tool0_m=mid.round(5).tolist(), gap_m=round(float(np.linalg.norm(d) - pad["thickness"]), 5),
                             closing_yaw_deg=round(math.degrees(yaw), 3), axis_tilt_deg=round(math.degrees(math.asin(min(1.0, abs(d[2]) / np.linalg.norm(d)))), 3)))
    finally:
        subprocess.run([sys.executable, os.path.join(HERE, "gripper.py"), "open", "--hold", "1.2"], capture_output=True)
        rclpy.shutdown()

    op = rows[0]
    stamp = time.strftime("%Y%m%d_%H%M%S"); os.makedirs(os.path.join(cell.WS, "calib"), exist_ok=True)
    rec = os.path.join(cell.WS, "calib", f"gripper_tf_{stamp}.json")
    run_io.write_json(rec, dict(method="robot model: /tf finger link poses + pad centre from the gripper CAD; TCP = pad-centre midpoint",
                                sweep=rows))
    lines = [f"{'finger_joint':>12} {'gap mm':>7} {'TCP x':>7} {'TCP y':>7} {'TCP z':>7} {'yaw deg':>8}"]
    lines += [f"{r['finger_joint_rad']:12.2f} {r['gap_m'] * 1000:7.1f} " + " ".join(f"{v * 1000:7.1f}" for v in r["tcp_tool0_m"])
              + f" {r['closing_yaw_deg']:8.2f}" for r in rows]
    lines.append(f"-> {rec}")
    if a.write:
        path = os.path.join(cell.WS, "config", "cell.yaml")
        bak = path + time.strftime(".%Y%m%d-%H%M%S.bak"); shutil.copy2(path, bak)
        txt = open(path).read()
        f3 = lambda v: "[" + ", ".join(f"{x:.5f}" for x in v) + "]"
        block = (f"  # --- FROM /tf (robot model: finger links + pad CAD), written by tools/gripper_tcp_tf.py --write; do not edit by hand\n"
                 f"  tool0_T_tcp:             # /tf {time.strftime('%Y-%m-%d %H:%M')} ({os.path.basename(rec)}), gripper open; pad-centre midpoint\n"
                 f"    xyz: {f3(op['tcp_tool0_m'])}\n    rpy: [0.0, 0.0, 0.0]\n"
                 f"  closing_axis_yaw_in_tool0: {math.radians(op['closing_yaw_deg']):.6f}   # rad, /tf\n"
                 f"  opening_table:           # /tf sweep: gap = inner pad faces; tcp_z = pad-centre midpoint depth below the flange\n"
                 f"    finger_joint_rad: [{', '.join(str(r['finger_joint_rad']) for r in rows)}]\n"
                 f"    gap_m:            {f3([r['gap_m'] for r in rows])}\n"
                 f"    tcp_z_m:          {f3([r['tcp_tool0_m'][2] for r in rows])}\n")
        new, n = re.subn(r"  # --- FROM /tf.*?(?=\n\n)", block.rstrip("\n"), txt, count=1, flags=re.S)
        if n != 1:
            raise SystemExit("cell.yaml: the '# --- FROM /tf' block was not found; nothing written")
        open(path, "w").write(new); lines.append(f"written to cell.yaml (backup {bak})")
    run_io.show(NAME, "OUT", lines)


if __name__ == "__main__":
    main()
