#!/usr/bin/env python3
"""REFEREE TOOL (scoring only). Compares script 4's objects with the simulator's ground truth.

The problem statement asks for perception error against Isaac's ground-truth poses. This is the
only place ground truth meets pipeline output; nothing here is ever fed back to the robot.
Ground truth is in Isaac's world frame; referee.world_T_base (cell.yaml) moves it into the base frame.
  IN : <run>/4_localize_<label>/objects.json, live /tf ground truth for each catalogue object's gt_frame
  OUT: <run>/score_<label>.json (per object: position error mm, yaw error deg) and a printout

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/referee_score.py [--run latest] [--label pick_overview]
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import cell, run_io, geometry as geo, kinematics as kin

NAME = "referee"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="latest"); ap.add_argument("--label"); a = ap.parse_args()
    cfg = cell.load(); run = run_io.resolve(a.run)
    loc = os.path.join(run, f"4_localize_{a.label}") if a.label else run_io.latest_step(run, "4_localize_")
    label = os.path.basename(loc).replace("4_localize_", "")
    objs = run_io.read_json(os.path.join(loc, "objects.json"))["objects"]
    frames = {o["gt_frame"]: name for name, o in cfg["objects"].items()}
    rclpy.init(); n = Node("referee_score"); gt = {}
    n.create_subscription(TFMessage, "/tf", lambda m: [gt.__setitem__(t.child_frame_id, t.transform) for t in m.transforms if t.child_frame_id in frames], 20)
    end = time.time() + 6
    while time.time() < end and len(gt) < len(frames):
        rclpy.spin_once(n, timeout_sec=0.05)
    rclpy.shutdown()
    run_io.show(NAME, "IN", [f"{loc}/objects.json: {len(objs)} object(s)", f"ground truth frames found: {sorted(gt)}"])
    base_T_world = np.linalg.inv(kin.T_from_cfg(cfg["referee"]["world_T_base"]))
    rows = []
    for o in objs:
        best = None
        for fr, tr in gt.items():
            Tw = np.eye(4); Tw[:3, :3] = kin.quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
            Tw[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
            Tb = base_T_world @ Tw; d = np.linalg.norm(Tb[:2, 3] - np.array(o["position"][:2]))
            if d < 0.1 and (best is None or d < best[0]):
                best = (d, fr, Tb)
        if best is None:
            rows.append(dict(id=o["id"], matched=None)); continue
        _, fr, Tb = best; e = (np.array(o["position"]) - Tb[:3, 3]) * 1000
        sym = cfg["objects"][frames[fr]].get("yaw_symmetry_deg", 360)
        rows.append(dict(id=o["id"], matched=fr, true_type=frames[fr], type_correct=o["type"] == frames[fr],
                         error_mm=e.round(2).tolist(), error_norm_mm=round(float(np.linalg.norm(e)), 2),
                         yaw_error_deg=round(float(geo.wrap(o["yaw_deg"] - geo.yaw_deg(Tb[:3, :3]), sym)), 2)))
    out = os.path.join(run, f"score_{label}.json"); run_io.write_json(out, dict(objects=rows))
    lines = [out]
    for r in rows:
        lines.append(f"#{r['id']}: no ground-truth object within 10 cm" if r.get("matched") is None else
                     f"#{r['id']} vs {r['matched']}: type {'correct' if r['type_correct'] else 'WRONG'}, error {r['error_mm']} mm "
                     f"(|e| {r['error_norm_mm']} mm), yaw {r['yaw_error_deg']} deg")
    run_io.show(NAME, "OUT", lines)


if __name__ == "__main__":
    main()
