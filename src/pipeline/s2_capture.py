#!/usr/bin/env python3
"""Script 2 - CAPTURE. Records one RGB-D frame and the robot's joint angles at that moment.

Raw sensor data only: no calibration, no detection. Waits until the arm is still, then takes only a
colour + depth pair whose timestamps are after that moment and within capture.max_pair_dt of each other.
  IN : (live) /rgb, /depth, /camera_info, /joint_states; a label (usually the capture pose name)
  OUT: <run>/2_capture_<label>/  rgb.png, depth.npy (float32 metres, inf = no return),
       camera_info.json (K, D, size), capture.json (joints at image time, how they were matched, stamps)

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/pipeline/s2_capture.py --label pick_overview [--run latest | --new-run]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import rclpy
import cell, run_io
from ros_io import RobotIO, stamp_s

NAME = "s2 capture"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="manual")
    ap.add_argument("--run", default=None, help="'latest' or a run name; default: a new run")
    a = ap.parse_args()
    cfg = cell.load()
    run = run_io.resolve(a.run) if a.run else run_io.new_run(a.label)
    run_io.show(NAME, "IN", ["live topics /rgb /depth /camera_info /joint_states", f"label {a.label}, run {os.path.basename(run)}"])

    rclpy.init(); io = RobotIO("s2_capture", camera=True)
    if not io.wait_for(lambda: io.q is not None and io.info is not None and io.rgb is not None, 15):
        raise SystemExit("no camera or joint states: is Isaac playing on domain 33?")
    if not io.wait_still(cfg["capture"]["max_joint_speed"]):
        raise SystemExit("arm did not stop moving; capture refused")
    # Only frames TAKEN after the arm was confirmed still, by their own timestamps (not by when they arrived:
    # a camera driver or a ROS queue can deliver an older frame late). Colour and depth paired by timestamp.
    still_stamp = io.hist[-1][0]                                  # joint-state time at which the arm was confirmed still
    max_dt = cfg["capture"]["max_pair_dt"]
    got = {}

    def ready():
        got["pair"] = io.fresh_pair(still_stamp, max_dt)
        return got["pair"] is not None

    if not io.wait_for(ready, 5):
        last = [f"{k} {stamp_s(m.header.stamp):.3f}" for k, m in (("depth", io.depth), ("rgb", io.rgb)) if m is not None]
        raise SystemExit(f"no colour+depth pair taken after the arm stopped (still at {still_stamp:.3f}; latest {', '.join(last)}; "
                         f"pairing tolerance {max_dt} s); capture refused")
    d, rgb = got["pair"]; info = io.info
    source, q = io.joints_at(d.header.stamp, cfg["capture"]["max_pose_age"])
    rclpy.shutdown()

    out = run_io.step_dir(run, f"2_capture_{a.label}")
    depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width).copy()
    img = np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.width, 3)
    np.save(os.path.join(out, "depth.npy"), depth)
    cv2.imwrite(os.path.join(out, "rgb.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    K = np.array(info.k).reshape(3, 3)
    run_io.write_json(os.path.join(out, "camera_info.json"), dict(width=info.width, height=info.height, K=K,
                      D=list(info.d), distortion_model=info.distortion_model, frame_id=info.header.frame_id))
    stamp = stamp_s(d.header.stamp)
    run_io.write_json(os.path.join(out, "capture.json"), dict(label=a.label, image_stamp=stamp, rgb_stamp=stamp_s(rgb.header.stamp),
                      still_since_stamp=still_stamp, joints=q,
                      joints_source=source, wall_time=time.strftime("%Y-%m-%d %H:%M:%S")))
    valid = np.isfinite(depth) & (depth > 0)
    run_io.show(NAME, "OUT", [f"{out}/",
                              f"rgb.png {img.shape[1]}x{img.shape[0]}, depth.npy {valid.mean()*100:.1f}% valid "
                              f"({depth[valid].min():.3f}..{depth[valid].max():.3f} m)",
                              f"camera_info.json fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}",
                              f"capture.json joints {np.round(q, 4).tolist()} rad ({source})",
                              f"depth taken {stamp - still_stamp:.3f} s after the arm was still; colour-depth gap {abs(stamp_s(rgb.header.stamp) - stamp) * 1000:.0f} ms"])


if __name__ == "__main__":
    main()
