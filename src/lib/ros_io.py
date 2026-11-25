"""ROS 2 helpers shared by the scripts that talk to the robot or the camera."""
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from cell import ARM_JOINTS


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RobotIO(Node):
    """Joint angles with timestamps and speed; optional camera frames; a joint command publisher."""

    def __init__(self, name, camera=False, command=False):
        super().__init__(name)
        self.hist = deque(maxlen=600)              # (stamp seconds, joint angles)
        self.q = None; self.speed = None; self._prev = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 20)
        self.rgb = self.depth = self.info = None; self.depth_arrival = 0.0
        self.rgb_recent = deque(maxlen=10)        # recent colour frames, to pair with a depth frame by timestamp
        if camera:
            self.create_subscription(Image, "/rgb", self._on_rgb, 10)
            self.create_subscription(Image, "/depth", self._on_depth, 2)
            self.create_subscription(CameraInfo, "/camera_info", lambda m: setattr(self, "info", m), 2)
        self.pub = self.create_publisher(JointState, "/joint_command", 10) if command else None

    def _on_rgb(self, m):
        self.rgb = m; self.rgb_recent.append(m)

    def _on_depth(self, m):
        self.depth = m; self.depth_arrival = time.time()

    def _on_js(self, m):
        if not all(j in m.name for j in ARM_JOINTS):
            return
        idx = [m.name.index(j) for j in ARM_JOINTS]
        q = np.array([m.position[i] for i in idx]); now = time.time()
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.hist.append((t, q.copy())); self.q = q
        # speed from position changes over the message stamps, not the velocity field: with the gripper squeezing a box,
        # Isaac reports ~0.03 rad/s arm velocity while the positions do not change (PT-5, 2026-10-23)
        if self._prev is not None and t > self._prev[0]:
            self.speed = float(np.max(np.abs(q - self._prev[1])) / (t - self._prev[0]))
        elif self._prev is not None and now - self._prev[2] > 1.0:
            self.speed = None                      # stamps not advancing (paused): unknown
        if self._prev is None or t > self._prev[0]:
            self._prev = (t, q, now)

    def spin_for(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not cond():
            rclpy.spin_once(self, timeout_sec=0.02)
        return cond()

    def wait_still(self, max_speed, hold=0.5, timeout=8.0):
        end = time.time() + timeout; since = None
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.speed is not None and self.speed < max_speed:
                since = since or time.time()
                if time.time() - since >= hold:
                    return True
            else:
                since = None
        return False

    def fresh_pair(self, not_before, max_dt):
        """(depth, rgb) taken at or after `not_before` (header-stamp seconds), the colour frame nearest the depth
        frame and within max_dt of it; None if there is no such pair yet."""
        if self.depth is None or stamp_s(self.depth.header.stamp) < not_before or not self.rgb_recent:
            return None
        td = stamp_s(self.depth.header.stamp)
        rgb = min(self.rgb_recent, key=lambda m: abs(stamp_s(m.header.stamp) - td))
        return (self.depth, rgb) if abs(stamp_s(rgb.header.stamp) - td) <= max_dt else None

    def joints_at(self, stamp, max_age):
        """Joint angles recorded closest to `stamp`; ('stamped', q) if within max_age, else ('latest-while-still', q)."""
        t_img = stamp.sec + stamp.nanosec * 1e-9
        t, q = min(self.hist, key=lambda s: abs(s[0] - t_img))
        if abs(t - t_img) <= max_age:
            return "joint angles stamped at the image time", q
        return "latest joint angles, arm still", self.hist[-1][1]
