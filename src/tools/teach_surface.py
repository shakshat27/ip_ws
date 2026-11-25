#!/usr/bin/env python3
"""Teach a resting surface (belt, table) by measuring it with the calibrated wrist camera.

From the current, still arm pose:
  1. depth pixels -> 3D points in the robot base frame (FK x calibrated hand-eye; nothing from the simulator)
  2. RANSAC: repeatedly fit a plane through 3 random points, keep the plane most points agree with
     (within 3 mm), preferring planes that are level (normal within 10 deg of vertical)
  3. refine that plane by least squares over all its inlier points
  4. repeat on the remaining points to find every level plane (floor, belt, ...); the one taught is the plane
     under the image centre, i.e. where the operator points the camera (or --plane N from the printed list)
  5. that plane's region in the image -> its largest connected patch -> the smallest rectangle around it
Writes centre, size, yaw and height to cell.yaml surfaces.<name> with --write.
Real cell: identical. Note the rectangle is the part of the surface the camera sees from this pose.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/teach_surface.py --name belt_a             # measure and report
  python3 ~/ip_ws/src/tools/teach_surface.py --name belt_a --write     # also store it (backup kept)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib"))

import argparse, math, os, shutil, time
import numpy as np, cv2, yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
import kinematics as kin

CFG = os.path.expanduser("~/ip_ws/config/cell.yaml")
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


class Grab(Node):
    def __init__(self):
        super().__init__("teach_surface"); self.depth = self.info = self.q = None; self.speed = None
        self.create_subscription(Image, "/depth", lambda m: setattr(self, "depth", m), 2)
        self.create_subscription(CameraInfo, "/camera_info", lambda m: setattr(self, "info", m), 2)
        self.create_subscription(JointState, "/joint_states", self._js, 10)

    def _js(self, m):
        if all(j in m.name for j in ARM):
            self.q = [m.position[m.name.index(j)] for j in ARM]
            if len(m.velocity) == len(m.name):
                self.speed = max(abs(m.velocity[m.name.index(j)]) for j in ARM)


def ransac_plane(P, iters=400, tol=0.003, max_tilt_deg=10.0, rng=np.random.default_rng(0)):
    best, best_n = None, 0
    cos_lim = math.cos(math.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n /= L
        if abs(n[2]) < cos_lim:                       # not level: not a resting surface
            continue
        k = int((np.abs((P - a) @ n) < tol).sum())
        if k > best_n:
            best, best_n = (n, a), k
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--write", action="store_true")
    ap.add_argument("--plane", type=int, help="index from the printed list, instead of the plane under the image centre")
    ap.add_argument("--bridge", type=int, default=41, help="px: gaps up to this wide join one surface (roller gaps); 0 = off")
    a = ap.parse_args(); cfg = yaml.safe_load(open(CFG))
    he = cfg["hand_eye"]["tool0_T_camera"]
    if he is None:
        print("hand-eye not calibrated yet"); return
    rclpy.init(); n = Grab(); t0 = time.time()
    while (n.depth is None or n.info is None or n.q is None) and time.time() - t0 < 15:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.depth is None or n.q is None:
        print("no depth / joint states: is Isaac playing?"); return
    if n.speed is not None and n.speed > cfg["capture"]["max_joint_speed"]:
        print("arm is moving: stop at a capture pose first"); return
    m = n.depth; q = n.q; rclpy.shutdown()
    H, W = m.height, m.width; d = np.frombuffer(m.data, np.float32).reshape(H, W)
    K = n.info.k; v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    ok = np.isfinite(d) & (d > cfg["detection"]["near_cutoff"])
    Z = np.where(ok, d, 0.0); P = np.stack([(u - K[2]) * Z / K[0], (v - K[5]) * Z / K[4], Z], -1)
    T = kin.fk(q, cfg["robot"]["dh"]) @ kin.T_from_cfg(he)
    B = P @ T[:3, :3].T + T[:3, 3]                              # every pixel in the base frame
    pts = B[ok]; sub = pts[np.random.default_rng(1).choice(len(pts), min(60000, len(pts)), replace=False)]
    planes, rest = [], np.ones(len(pts), bool)
    for _ in range(4):                                          # sequential RANSAC: peel off one level plane at a time
        cand = pts[rest]
        if len(cand) < 0.03 * len(pts):
            break
        sub = cand[np.random.default_rng(len(planes)).choice(len(cand), min(60000, len(cand)), replace=False)]
        found = ransac_plane(sub)
        if found is None:
            break
        nrm, a0 = found
        inl = rest & (np.abs((pts - a0) @ nrm) < 0.003)
        if inl.sum() < 0.03 * len(pts):
            break
        c = pts[inl].mean(0); Q = pts[inl] - c
        w, V = np.linalg.eigh(Q.T @ Q)                           # least-squares refit: normal = direction of least spread
        nrm = V[:, 0] * np.sign(V[2, 0])
        planes.append((c, nrm, inl)); rest &= ~inl
    if not planes:
        print("no level plane found"); return
    full = np.zeros((H, W), np.int32) - 1
    idx = np.flatnonzero(ok.ravel())
    for k_, (_, _, inl) in enumerate(planes):
        full.ravel()[idx[inl]] = k_
    centre_plane = int(full[H // 2, W // 2])
    print("level planes found (base frame):")
    for k_, (c, nrm, inl) in enumerate(planes):
        print(f"  [{k_}] height {c[2]:+.4f} m, {inl.sum():,} points{'   <- under the image centre' if k_ == centre_plane else ''}")
    pick = a.plane if a.plane is not None else centre_plane
    if pick < 0:
        print("the image centre is not on a level plane: point the camera at the surface, or pass --plane N"); return
    c, nrm, inl = planes[pick]
    resid = (pts[inl] - c) @ nrm
    mask = (full == pick).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # a roller conveyor's surface is the roller tops: separate strips with gaps where depth sees the floor; bridge the
    # gaps so the rollers form one patch, but fit the rectangle only through the plane's own points
    if a.bridge > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((a.bridge, a.bridge), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = int(lab[H // 2, W // 2]) if a.plane is None and lab[H // 2, W // 2] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    region = B[(lab == big) & (full == pick)]
    (mx, my), (L1, L2), ang = cv2.minAreaRect((region[:, :2] * 1000).astype(np.float32))
    if L1 < L2:                                                  # report size as (along, across) with along = longer side
        L1, L2, ang = L2, L1, ang + 90.0
    yaw = math.radians(((ang + 90) % 180) - 90)
    zc = c[2] - (nrm[0] * (mx / 1000 - c[0]) + nrm[1] * (my / 1000 - c[1])) / nrm[2]   # plane height at the rectangle centre
    tilt = math.degrees(math.acos(min(1.0, abs(nrm[2]))))
    print(f"taught plane [{pick}]: {inl.sum():,} points within 3 mm; flatness (std) {resid.std() * 1000:.2f} mm; tilt {tilt:.2f} deg")
    print(f"surface '{a.name}' (seen part): centre ({mx / 1000:+.4f}, {my / 1000:+.4f}, {zc:+.4f}) m, size {L1 / 1000:.3f} x {L2 / 1000:.3f} m, yaw {math.degrees(yaw):+.2f} deg (base frame)")
    if a.write:
        bak = CFG + time.strftime(".%Y%m%d-%H%M%S.bak"); shutil.copy2(CFG, bak)          # operator rule: <file>.<ext>.<stamp>.bak next to it
        txt = open(CFG).read(); key = f"  {a.name}:"
        i = txt.index(key); j = txt.index("\n", i)
        while txt[j + 1:j + 1 + 4] == "    " or txt[j + 1:j + 1 + 26].strip().startswith("#"):
            j = txt.index("\n", j + 1)
        block = (f"  {a.name}:                 # TAUGHT {time.strftime('%Y-%m-%d %H:%M')} by camera: RANSAC plane, flatness {resid.std()*1000:.2f} mm, tilt {tilt:.2f} deg; seen part only\n"
                 f"    centre: [{mx/1000:.4f}, {my/1000:.4f}, {zc:.4f}]\n    size: [{L1/1000:.4f}, {L2/1000:.4f}]\n    yaw: {yaw:.5f}")
        open(CFG, "w").write(txt[:i] + block + txt[j:])
        print(f"written to cell.yaml (backup {bak})")


if __name__ == "__main__":
    main()
