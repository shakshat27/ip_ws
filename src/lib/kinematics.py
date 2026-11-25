#!/usr/bin/env python3
"""UR10 forward kinematics: joint angles -> pose of the flange (tool0) in the robot base frame.

This is how the camera knows where it is, without asking the simulator:
    base_T_camera = FK(joint angles) @ tool0_T_camera
FK uses the UR10's Denavit-Hartenberg (DH) parameters, published by Universal Robots.
tool0_T_camera is the camera's fixed offset from the flange: nominal from the bracket
design, later replaced by a hand-eye calibration. On a real UR10 the same code runs with
that robot's own calibrated DH values (ur_calibration extracts them from the controller).

Frames (UR convention):
  base  : robot base. Z up through joint 1, X/Y as on the teach pendant's "Base" view.
  tool0 : flange centre. Z points out of the flange along the tool.
"""
import math
import numpy as np

# Universal Robots UR10 (CB-series) nominal DH parameters, metres and radians.
UR10_DH = {
    "d":     [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
    "a":     [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0],
    "alpha": [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
}


def dh_matrix(a, alpha, d, theta):
    """Standard DH transform from frame i-1 to frame i."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def fk(q, dh=UR10_DH):
    """Six joint angles (rad) -> 4x4 pose of tool0 in the base frame."""
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_matrix(dh["a"][i], dh["alpha"][i], dh["d"][i], q[i])
    return T


def rpy_to_R(roll, pitch, yaw):
    """Fixed-axis X then Y then Z rotations (the URDF / ROS convention): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def R_to_rpy(R):
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def make_T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_to_R(*rpy); T[:3, 3] = xyz
    return T


def T_from_cfg(entry):
    return make_T(entry["xyz"], entry["rpy"])


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    # Self-test against values Universal Robots documents for the UR10.
    T0 = fk([0, 0, 0, 0, 0, 0])
    expect0 = np.array([-0.612 - 0.5723, -(0.163941 + 0.0922), 0.1273 - 0.1157])   # arm stretched along -X
    e0 = np.linalg.norm(T0[:3, 3] - expect0) * 1000
    # "Ready" pose: shoulder lift -90, elbow +90, wrist1 -90, wrist2 -90. Tool points straight down.
    Tr = fk([0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0])
    down = Tr[:3, 2] @ np.array([0, 0, -1])
    # Orthonormal rotation for random poses
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(1000):
        R = fk(rng.uniform(-math.pi, math.pi, 6))[:3, :3]
        worst = max(worst, np.abs(R.T @ R - np.eye(3)).max())
    print(f"zero pose  tool0 at {T0[:3, 3].round(4)} m, expected {expect0.round(4)}: error {e0:.3f} mm")
    print(f"ready pose tool0 at {Tr[:3, 3].round(4)} m, tool axis . down = {down:.4f} (1 = pointing straight down)")
    print(f"rotation orthonormal over 1000 random poses: worst deviation {worst:.1e}")
    ok = e0 < 0.01 and down > 0.9999 and worst < 1e-12
    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
