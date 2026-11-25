#!/usr/bin/env python3
"""Eye-in-hand calibration: find where the camera sits on the flange (tool0_T_camera).

Real-cell procedure, using only what a real robot and camera provide:
  - joint angles (robot)  -> forward kinematics -> base_T_tool0 at each pose
  - colour image + lens model (camera) -> ChArUco board pose -> camera_T_board at each pose
The board stays fixed and its position need not be known. For every pair of poses,
    base_T_tool0(i) @ X @ camera_T_board(i)  =  base_T_board   (the same for all i)
and cv2.calibrateHandEye solves for X = tool0_T_camera.

Quality is judged without the simulator: with the solved X, every pose must put the board at
the same place in the base frame. The spread of those positions is the calibration's precision.

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/handeye_calibrate.py           # run, report, save raw data in ~/ip_ws/calib/
  python3 ~/ip_ws/src/tools/handeye_calibrate.py --write   # also write the result into cell.yaml (backup kept)
  python3 ~/ip_ws/src/tools/handeye_calibrate.py --from ~/ip_ws/calib/handeye_<stamp>.json [--write]   # re-solve saved data, no motion
Before running: the board must lie flat, fixed, where the camera sees it from capture pose pick_overview.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib"))

import argparse, json, math, os, shutil, time
import numpy as np, cv2, yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
import kinematics as kin
from charuco import board_from

CFG = os.path.expanduser("~/ip_ws/config/cell.yaml")
OUT = os.path.expanduser("~/ip_ws/calib")
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Offsets (rad) from the capture pose: rotations about several axes are what make the problem solvable.
OFFSETS = [
    [0, 0, 0, 0, 0, 0],
    [0.10, 0, 0, 0, 0, 0.35], [-0.10, 0, 0, 0, 0, -0.35],
    [0, 0, 0, 0.22, 0, 0], [0, 0, 0, -0.22, 0, 0],
    [0, 0, 0, 0, 0.22, 0], [0, 0, 0, 0, -0.22, 0],
    [0, -0.06, 0.10, 0.15, 0.12, 0.5], [0, -0.06, 0.10, -0.15, -0.12, -0.5],
    [0.08, 0.05, -0.08, 0.18, -0.15, 0.2], [-0.08, 0.05, -0.08, -0.18, 0.15, -0.2],
    [0.05, 0, 0, 0.12, 0.18, 0.8], [-0.05, 0, 0, -0.12, -0.18, -0.8],
    [0, 0.04, -0.06, 0.25, 0.1, -0.6], [0, 0.04, -0.06, -0.25, -0.1, 0.6],
]
METHODS = {"Tsai": cv2.CALIB_HAND_EYE_TSAI, "Park": cv2.CALIB_HAND_EYE_PARK, "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
           "Andreff": cv2.CALIB_HAND_EYE_ANDREFF, "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS}
# --close: look-at poses around the board (tilt deg, azimuth deg, roll deg, distance m). Closer and more tilted than
# OFFSETS: the 2026-10-12 set (about 0.6 m, small tilts) left a 6 mm error along the viewing axis (PC-1).
CLOSE = ([(0, 0, r, 0.40) for r in (0, 25, -25)]
         + [(15, a, (-20, 0, 20)[k % 3], (0.38, 0.42)[k % 2]) for k, a in enumerate(range(0, 360, 45))]
         + [(25, a + 22.5, (20, -20, 0)[k % 3], 0.40) for k, a in enumerate(range(0, 360, 45))])


def rot(axis, ang):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * ang)[0]


def close_targets(cfg, q0, X, cam_T_board, K, size, spec):
    """Joint targets whose camera looks at the board centre from CLOSE's viewpoints. The board is located once from the
    capture pose with the current hand-eye X (its error only shifts the aim point by millimetres); flange poses come
    from X, joints from the RoboDK backend's IK polished on our DH. Skipped: no IK, a configuration flip from the
    capture pose (> 1.5 rad on a joint), or a board corner that would fall outside the image."""
    from planning import get_planner
    ik = get_planner("robodk", cfg).ik
    T0 = kin.fk(q0, cfg["robot"]["dh"]); C0 = T0 @ X; B = C0 @ cam_T_board
    w, h = spec["squares"][0] * spec["square"], spec["squares"][1] * spec["square"]
    corners_b = [B @ np.array([x, y, 0, 1.0]) for x in (0, w) for y in (0, h)]
    centre = (B @ np.array([w / 2, h / 2, 0, 1.0]))[:3]
    u0 = C0[:3, 3] - centre; u0 /= np.linalg.norm(u0)
    e1 = C0[:3, 0] - (C0[:3, 0] @ u0) * u0; e1 /= np.linalg.norm(e1); e2 = np.cross(u0, e1)
    out = []
    for tilt, az, roll, dist in CLOSE:
        u = rot(math.cos(math.radians(az)) * e1 + math.sin(math.radians(az)) * e2, math.radians(tilt)) @ u0 if tilt else u0
        z = -u; x = C0[:3, 0] - (C0[:3, 0] @ z) * z; x /= np.linalg.norm(x)
        x = rot(z, math.radians(roll)) @ x; y = np.cross(z, x)
        C = np.eye(4); C[:3, :3] = np.c_[x, y, z]; C[:3, 3] = centre + dist * u
        Ci = np.linalg.inv(C)
        uv = [K @ (Ci @ p)[:3] for p in corners_b]
        if any(p[2] <= 0 or not (40 < p[0] / p[2] < 2 * K[0, 2] - 40 and 40 < p[1] / p[2] < 2 * K[1, 2] - 40) for p in uv):
            out.append((tilt, az, roll, dist, None, "board would leave the image")); continue
        q = ik(C @ np.linalg.inv(X), q0)
        if q is None:
            out.append((tilt, az, roll, dist, None, "no IK")); continue
        if np.max(np.abs(q - q0)) > 1.5:
            out.append((tilt, az, roll, dist, None, "configuration flip")); continue
        out.append((tilt, az, roll, dist, q, ""))
    return out, centre


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Cal(Node):
    def __init__(self):
        super().__init__("handeye_calibrate")
        self.rgb = self.info = self.q = None; self.rgb_t = 0.0; self.speed = None; self._prev = None
        self.create_subscription(Image, "/rgb", self._on_rgb, 2)
        self.create_subscription(CameraInfo, "/camera_info", lambda m: setattr(self, "info", m), 2)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.pub = self.create_publisher(JointState, "/joint_command", 10)

    def _on_rgb(self, m):
        self.rgb = m; self.rgb_t = time.time()

    def _on_js(self, m):
        if not all(j in m.name for j in ARM):
            return
        q = np.array([m.position[m.name.index(j)] for j in ARM])
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        # speed from position changes over the stamps, not the velocity field (Isaac's can be non-zero at rest: PT-5)
        if self._prev is not None and st > self._prev[0]:
            self.speed = float(np.max(np.abs(q - self._prev[1])) / (st - self._prev[0]))
        if self._prev is None or st > self._prev[0]:
            self._prev = (st, q)
        self.q = q

    def spin_for(self, secs):
        t = time.time()
        while time.time() - t < secs:
            rclpy.spin_once(self, timeout_sec=0.01)

    def move(self, target, max_speed=0.35):
        start = self.q.copy(); target = np.array(target)
        dur = max(1.5, float(np.max(np.abs(target - start))) / max_speed * 1.875)   # min-jerk peak speed = 1.875 x mean
        t0 = time.time(); msg = JointState(); msg.name = ARM
        while True:
            s = min(1.0, (time.time() - t0) / dur)
            msg.position = list(start + min_jerk(s) * (target - start)); self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0); time.sleep(1 / 60)
            if s >= 1.0:
                break
        for _ in range(30):
            self.pub.publish(msg); self.spin_for(1 / 60)

    def settle(self, still=0.005, hold=0.6, timeout=6.0):
        t0 = time.time(); since = None
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.speed is not None and self.speed < still:
                since = since or time.time()
                if time.time() - since >= hold:
                    return True
            else:
                since = None
        return False

    def fresh_frame(self):
        t = time.time()
        while self.rgb_t <= t:
            rclpy.spin_once(self, timeout_sec=0.01)
        return self.rgb, self.q.copy()


def detect_board(img_msg, info, board, dic, min_corners=10):
    img = np.frombuffer(img_msg.data, np.uint8).reshape(img_msg.height, img_msg.width, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    K = np.array(info.k).reshape(3, 3); D = np.array(info.d) if len(info.d) else np.zeros(5)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dic)
    if ids is None or len(ids) < 4:
        return None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=K, distCoeffs=D)
    if n is None or n < min_corners:
        return None
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D, None, None)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(board.chessboardCorners[ci.ravel()], rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - cc.reshape(-1, 2)) ** 2, 1))))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
    return T, int(n), rms


def consistency(X, base_T_tool, cam_T_board):
    """With X, every pose predicts base_T_board. Their spread is the calibration's precision."""
    P = [bt @ X @ cb for bt, cb in zip(base_T_tool, cam_T_board)]
    pos = np.array([p[:3, 3] for p in P]); mean = pos.mean(0)
    d_pos = np.linalg.norm(pos - mean, axis=1) * 1000
    R0 = P[0][:3, :3]
    d_rot = [math.degrees(math.acos(max(-1, min(1, (np.trace(R0.T @ p[:3, :3]) - 1) / 2)))) for p in P]
    return float(np.sqrt(np.mean(d_pos ** 2))), float(d_pos.max()), float(np.std(d_rot))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    ap.add_argument("--from", dest="src", help="re-solve a saved run instead of moving the robot")
    ap.add_argument("--close", action="store_true", help="look-at poses about 0.4 m from the board with 15-25 deg tilts (CLOSE), "
                                                          "aimed with the current hand-eye, instead of the joint OFFSETS")
    ap.add_argument("--dry", action="store_true", help="with --close: list the poses and their IK, move nothing"); a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG)); spec = cfg["calibration"]["charuco"]
    board, dic = board_from(spec); dh = cfg["robot"]["dh"]
    if a.src:
        raw = os.path.expanduser(a.src); data = json.load(open(raw)); samples = data["samples"]; dh = data["dh"]
        print(f"re-solving {len(samples)} saved poses from {raw}")
        return solve(samples, dh, raw, a.write)
    q0 = np.array(cfg["capture_poses"]["pick_overview"]["joints"])
    rclpy.init(); n = Cal(); t0 = time.time()
    while (n.q is None or n.info is None or n.rgb is None) and time.time() - t0 < 15:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.q is None or n.info is None or n.rgb is None:
        print("no joint states / camera: is Isaac playing on domain 33?"); return
    targets = [q0 + np.array(off) for off in OFFSETS]
    if a.close:
        import cell
        ccfg = cell.load(); X0 = kin.T_from_cfg(ccfg["hand_eye"]["tool0_T_camera"])
        n.move(q0)
        if not n.settle():
            print("arm did not settle at the capture pose"); rclpy.shutdown(); return
        img, _ = n.fresh_frame(); det = detect_board(img, n.info, board, dic)
        if det is None:
            print("board not found from the capture pose"); rclpy.shutdown(); return
        K = np.array(n.info.k).reshape(3, 3)
        plan, centre = close_targets(ccfg, q0, X0, det[0], K, None, spec)
        print(f"board centre (current hand-eye) {np.round(centre, 4).tolist()} m in the base frame")
        for k, (tilt, az, roll, dist, q, why) in enumerate(plan):
            print(f"  pose {k:2d}: tilt {tilt:4.1f} deg, azimuth {az:5.1f}, roll {roll:+3d}, {dist:.2f} m -> "
                  + (f"joints {np.round(q, 3).tolist()}" if q is not None else f"skipped: {why}"))
        targets = [q for *_, q, _ in plan if q is not None]
        if a.dry:
            print(f"{len(targets)} usable poses (dry run, nothing moved)"); rclpy.shutdown(); return
    samples = []
    for k, target in enumerate(targets):
        n.move(target)
        if not n.settle():
            print(f"pose {k:2d}: arm did not settle, skipped"); continue
        img, q = n.fresh_frame()
        det = detect_board(img, n.info, board, dic)
        if det is None:
            print(f"pose {k:2d}: board not found (or under 10 corners), skipped"); continue
        cam_T_board, corners, rms = det
        samples.append(dict(q=q.tolist(), cam_T_board=cam_T_board.tolist(), corners=corners, reproj_px=rms))
        print(f"pose {k:2d}: {corners:2d} corners, reprojection {rms:.2f} px, board {cam_T_board[2, 3]:.3f} m from camera")
    n.move(q0); rclpy.shutdown()

    os.makedirs(OUT, exist_ok=True)
    raw = f"{OUT}/handeye_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(dict(board=spec, dh=dh, samples=samples), open(raw, "w"), indent=1)
    print(f"\n{len(samples)} usable poses; raw data saved to {raw}")
    solve(samples, dh, raw, a.write)


def solve(samples, dh, raw, write):
    if len(samples) < 6:
        print("too few poses: move the board so the camera sees it from the capture pose, then rerun"); return

    base_T_tool = [kin.fk(s["q"], dh) for s in samples]
    cam_T_board = [np.array(s["cam_T_board"]) for s in samples]
    results = {}
    for name, m in METHODS.items():
        R, t = cv2.calibrateHandEye([T[:3, :3] for T in base_T_tool], [T[:3, 3] for T in base_T_tool],
                                    [T[:3, :3] for T in cam_T_board], [T[:3, 3] for T in cam_T_board], method=m)
        X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t.ravel()
        results[name] = (X,) + consistency(X, base_T_tool, cam_T_board)
    print("\nmethod       camera offset in tool0 (mm)          board spread: rms / max (mm), rotation (deg)")
    for name, (X, rms, mx, rot) in results.items():
        print(f"{name:11s}  {np.round(X[:3, 3] * 1000, 1).tolist()!s:32s}  {rms:6.2f} / {mx:6.2f}   {rot:.3f}")
    best = min(results, key=lambda k: results[k][1]); X = results[best][0]
    xyz = [round(float(v), 5) for v in X[:3, 3]]; rpy = [round(float(v), 6) for v in kin.R_to_rpy(X[:3, :3])]
    print(f"\nchosen: {best} (smallest spread {results[best][1]:.2f} mm). tool0_T_camera xyz {xyz} m, rpy {rpy} rad")
    # check: where the board surface sits in the base frame, with the new and the current calibration (heights matter: PC-1)
    cfg = yaml.safe_load(open(CFG)); spec = cfg["calibration"]["charuco"]
    mid = np.array([spec["squares"][0] * spec["square"] / 2, spec["squares"][1] * spec["square"] / 2, 0, 1.0])
    old = kin.T_from_cfg(cfg["hand_eye"]["tool0_T_camera"])
    for label, Xc in (("new", X), ("current", old)):
        P = np.array([(bt @ Xc @ cb @ mid)[:3] for bt, cb in zip(base_T_tool, cam_T_board)])
        print(f"board centre with the {label:7s} hand-eye: {np.round(P.mean(0) * 1000, 1).tolist()} mm (z spread {P[:, 2].std() * 1000:.2f} mm)")
    print(f"camera mount change: {np.round((X[:3, 3] - old[:3, 3]) * 1000, 2).tolist()} mm")
    if write:
        bak = CFG + time.strftime(".%Y%m%d-%H%M%S.bak"); shutil.copy2(CFG, bak)          # operator rule: <file>.<ext>.<stamp>.bak next to it
        txt = open(CFG).read()
        i = txt.index("  tool0_T_camera:"); j = txt.index("\n\n", i)
        block = (f"  tool0_T_camera:          # CALIBRATED {time.strftime('%Y-%m-%d %H:%M')}: eye-in-hand, ChArUco, {best}, "
                 f"{len(samples)} poses, board spread {results[best][1]:.2f} mm rms ({os.path.basename(raw)})\n"
                 f"    xyz: {xyz}\n    rpy: {rpy}")
        open(CFG, "w").write(txt[:i] + block + txt[j:])
        print(f"written to cell.yaml (backup {bak})")


if __name__ == "__main__":
    main()
