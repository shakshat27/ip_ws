#!/usr/bin/env python3
"""Step 2: open / close the Robotiq 2F-85 through the same /joint_command path
the arm uses.  Only finger_joint is driven; the other five finger joints mimic it.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/gripper.py close            # finger_joint -> 0.70 rad
  python3 ~/ip_ws/src/tools/gripper.py open             # finger_joint -> 0.00 rad
  python3 ~/ip_ws/src/tools/gripper.py --pos 0.4        # any target, rad (0 open .. ~0.8 closed)
  python3 ~/ip_ws/src/tools/gripper.py close --expect-hold   # exit 0 if the fingers stop short (holding), 2 if not
"""
import argparse, math, sys, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGER = "finger_joint"
PRESETS = {"open": 0.0, "close": 0.70}


class Gripper(Node):
    def __init__(self):
        super().__init__("gripper")
        self.pos = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)

    def _on_js(self, msg):
        if FINGER in msg.name:
            self.pos = msg.position[msg.name.index(FINGER)]

    def wait(self, timeout=5.0):
        t0 = time.time()
        while self.pos is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pos is not None

    def command(self, target, hold, rate):
        start = self.pos
        msg = JointState(); msg.name = [FINGER]; msg.position = [target]
        t0 = time.time(); settled_at = None
        while time.time() - t0 < hold:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            if settled_at is None and abs(self.pos - target) < 0.02:
                settled_at = time.time() - t0
            time.sleep(1.0 / rate)
        err = self.pos - target
        self.get_logger().info(f"{FINGER}: {start:+.3f} -> {self.pos:+.3f} rad (target {target:+.3f}), "
                               f"error {math.degrees(err):+.2f} deg, "
                               + (f"settled after {settled_at:.2f} s" if settled_at is not None else "did NOT settle"))
        return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", choices=PRESETS.keys())
    ap.add_argument("--pos", type=float, help="finger_joint target in rad")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to keep commanding")
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--expect-hold", action="store_true",
                    help="grasp check: exit 0 if the fingers stopped more than 0.03 rad short of the target (an object between them), 2 if not")
    ap.add_argument("--expect-rad", type=float, default=None,
                    help="with --expect-hold: where the fingers should stop on this object (from its width); stopping further "
                         "than --tol from it means blocked by something else, exit 3")
    ap.add_argument("--tol", type=float, default=0.05, help="rad, tolerance for --expect-rad (about 5 mm of gap)")
    a = ap.parse_args()
    if a.pos is None and a.action is None:
        ap.error("give 'open', 'close' or --pos")
    target = a.pos if a.pos is not None else PRESETS[a.action]
    rclpy.init()
    g = Gripper()
    if not g.wait():
        print("no finger_joint in /joint_states within 5 s: is Isaac playing on domain 33?", file=sys.stderr)
        rclpy.shutdown(); sys.exit(1)
    err = g.command(target, a.hold, a.rate)
    rclpy.shutdown()
    if a.expect_hold:
        pos = target + err
        if abs(err) <= 0.03:
            verdict, code = "EMPTY", 2
        elif a.expect_rad is not None and abs(pos - a.expect_rad) > a.tol:
            verdict, code = f"BLOCKED (stopped at {pos:.3f} rad, the object needs about {a.expect_rad:.3f})", 3
        else:
            verdict, code = "HOLDING", 0
        print(f"grasp check: {verdict} ({FINGER} {math.degrees(err):+.1f} deg from the target)", flush=True)
        sys.exit(code)


if __name__ == "__main__":
    main()
