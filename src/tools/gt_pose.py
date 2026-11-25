#!/usr/bin/env python3
"""REFEREE TOOL (scoring only, never an input to the robot). Read the ground-truth pose of any object Isaac publishes on /tf.
This is the "truth" that perception will later be scored against.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/gt_pose.py              # list every frame Isaac publishes
  python3 ~/ip_ws/src/tools/gt_pose.py box          # pose of the frame named 'box', in world
  python3 ~/ip_ws/src/tools/gt_pose.py box --watch  # keep printing (drop or push the box and watch)

Isaac names each frame after its prim, and 'World' is the parent unless the graph says otherwise.
"""
import argparse, math, time
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def yaw_pitch_roll(q):
    """Quaternion (x, y, z, w) -> yaw, pitch, roll in degrees (Z-Y-X convention)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return [math.degrees(a) for a in (yaw, pitch, roll)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", nargs="?", help="child frame to report; omit to list all")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    rclpy.init(); node = Node("gt_pose"); latest = {}
    node.create_subscription(TFMessage, "/tf", lambda m: [latest.__setitem__(t.child_frame_id, t) for t in m.transforms], 10)
    t0 = time.time()
    while time.time() - t0 < 3:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not latest:
        print("nothing on /tf: is the transform tree node in the graph and Isaac playing on domain 33?"); return
    if a.frame is None:
        for name, t in sorted(latest.items()):
            p = t.transform.translation
            print(f"{t.header.frame_id:>12s} -> {name:30s} ({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}) m")
        return
    last = None
    while True:
        rclpy.spin_once(node, timeout_sec=0.05)
        t = latest.get(a.frame)
        if t is None:
            print(f"no frame '{a.frame}'. Known: {', '.join(sorted(latest))}"); break
        p, q = t.transform.translation, t.transform.rotation
        yaw, pitch, roll = yaw_pitch_roll(q)
        line = (f"{a.frame} in {t.header.frame_id}: position ({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f}) m  "
                f"yaw {yaw:+7.2f}  pitch {pitch:+6.2f}  roll {roll:+6.2f} deg")
        if line != last:
            print(line); last = line
        if not a.watch:
            break
        time.sleep(0.2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
