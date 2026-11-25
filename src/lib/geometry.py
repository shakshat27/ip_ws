"""Shared geometry: back-projection, camera pose from the robot, angle helpers."""
import math
import numpy as np
import kinematics as kin


def backproject(depth, K, near=0.0):
    """Depth image (H x W, metres along the optical axis) -> H x W x 3 points in the camera's optical
    frame (X right, Y down, Z forward) and a validity mask. Pinhole model: X = (u - cx) Z / fx."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    valid = np.isfinite(depth) & (depth > near)
    Z = np.where(valid, depth, 0.0).astype(np.float32)
    return np.stack([(u - K[0][2]) * Z / K[0][0], (v - K[1][2]) * Z / K[1][1], Z], -1), valid


def base_T_camera(joints, cfg):
    """Camera pose in the robot base frame from the robot alone: FK(joint angles) x calibrated hand-eye."""
    return kin.fk(joints, cfg["robot"]["dh"]) @ kin.T_from_cfg(cfg["hand_eye"]["tool0_T_camera"])


def to_frame(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def in_rectangle(P, surf):
    """Boolean mask: points whose x, y fall inside a taught rectangle (centre, size, yaw)."""
    c, s = math.cos(surf.get("yaw", 0.0)), math.sin(surf.get("yaw", 0.0))
    dx, dy = P[..., 0] - surf["centre"][0], P[..., 1] - surf["centre"][1]
    return (np.abs(c * dx + s * dy) <= surf["size"][0] / 2) & (np.abs(-s * dx + c * dy) <= surf["size"][1] / 2)


def wrap(a, period):
    """Smallest signed difference for an angle (deg) that repeats every `period` degrees."""
    return (a + period / 2) % period - period / 2


def yaw_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))
