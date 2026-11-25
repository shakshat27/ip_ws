#!/usr/bin/env python3
"""Does depth agree with colour? Diagnoses the camera height error (PC-1) without any simulator data.

One capture of the ChArUco board, measured twice in the camera frame:
  colour: board pose from the corners and the lens model (cv2 ChArUco)  -> plane of the board surface
  depth : plane fitted (least squares) to the depth pixels inside the board outline
Compared along the viewing axis at the board centre, and as the angle between the two planes.
  agree (about 1 mm or less)  -> depth and colour line up; a height error comes from the hand-eye -> re-calibrate
  differ (about 6 mm)         -> depth is offset from colour (alignment / depth scale), not the hand-eye
Also reports the board centre's height in the robot base frame through the current hand-eye, from both.
  IN : capture folders (rgb.png, depth.npy, camera_info.json, capture.json) or --fresh N (s2 captures now)
  OUT: printed table; ~/ip_ws/calib/depth_vs_colour_<stamp>.json

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/check_depth_vs_colour.py --fresh 3
  python3 ~/ip_ws/src/tools/check_depth_vs_colour.py --capture <run>/2_capture_x [...]
"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import cell, run_io, geometry as geo, kinematics as kin
from charuco import board_from

NAME = "depth vs colour"
PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")


def measure(cap, board, dic, cfg, he):
    rgb = cv2.imread(os.path.join(cap, "rgb.png")); gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    depth = np.load(os.path.join(cap, "depth.npy")); info = run_io.read_json(os.path.join(cap, "camera_info.json"))
    K = np.array(info["K"], float).reshape(3, 3); D = np.array(info.get("D") or [0] * 5, float)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dic)
    if ids is None or len(ids) < 4:
        raise RuntimeError(f"{cap}: board not found (markers: {0 if ids is None else len(ids)})")
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=K, distCoeffs=D)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D, None, None)
    if not ok or n < 10:
        raise RuntimeError(f"{cap}: board pose failed ({n} corners)")
    R, _ = cv2.Rodrigues(rvec); t = tvec.ravel()
    n_col = R[:, 2]                                                  # board normal (camera frame)
    # board centre on the surface (chessboard corners span the inner area; their mean is the centre)
    centre = R @ board.chessboardCorners[ci.ravel()].mean(0) + t
    # depth plane inside the detected corners' outline, shrunk 15 % to stay off the edges
    hull = cv2.convexHull(cc.reshape(-1, 2).astype(np.float32)); m = hull.reshape(-1, 2).mean(0)
    hull = ((hull.reshape(-1, 2) - m) * 0.85 + m).astype(np.int32)
    mask = np.zeros(depth.shape, np.uint8); cv2.fillConvexPoly(mask, hull, 1)
    pts, valid = geo.backproject(depth, K)
    P = pts[(mask > 0) & valid]
    c0 = P.mean(0); _, _, vt = np.linalg.svd(P - c0); n_dep = vt[2] * np.sign(vt[2] @ n_col)
    resid = (P - c0) @ n_dep
    # along the ray through the colour board centre: where does the depth plane cut it?
    ray = centre / np.linalg.norm(centre)
    s = (c0 @ n_dep) / (ray @ n_dep); hit = s * ray
    along = float(np.linalg.norm(hit) - np.linalg.norm(centre))         # + = depth sees the board further away
    ang = float(np.degrees(np.arccos(min(1.0, abs(n_col @ n_dep)))))
    q = run_io.read_json(os.path.join(cap, "capture.json"))["joints"]
    B = kin.fk(q, cfg["robot"]["dh"]) @ he
    z_col, z_dep = (B @ np.append(centre, 1))[2], (B @ np.append(hit, 1))[2]
    return dict(capture=os.path.relpath(cap, cell.RUNS), corners=int(n), board_distance_m=round(float(np.linalg.norm(centre)), 4),
                depth_minus_colour_mm=round(along * 1000, 2), plane_angle_deg=round(ang, 3), depth_plane_rms_mm=round(float(resid.std() * 1000), 3),
                depth_pixels=int(len(P)), board_z_base_colour_m=round(float(z_col), 5), board_z_base_depth_m=round(float(z_dep), 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", nargs="*", default=[]); ap.add_argument("--fresh", type=int, default=0)
    a = ap.parse_args()
    cfg = cell.load(); he = kin.T_from_cfg(cell.require(cfg, "hand_eye", "tool0_T_camera"))
    board, dic = board_from(cell.require(cfg, "calibration", "charuco"))
    caps = list(a.capture)
    if a.fresh:
        run = run_io.new_run("depthcheck"); rn = os.path.basename(run)
        for k in range(a.fresh):
            if subprocess.run([sys.executable, os.path.join(PIPE, "s2_capture.py"), "--label", f"board{k}", "--run", rn]).returncode != 0:
                raise SystemExit("capture failed")
            caps.append(os.path.join(run, f"2_capture_board{k}")); time.sleep(0.3)
    if not caps:
        raise SystemExit("give --capture folders or --fresh N")
    run_io.show(NAME, "IN", [f"{len(caps)} capture(s); board {cfg['calibration']['charuco']['squares']} x {cfg['calibration']['charuco']['square']} m",
                             "colour: ChArUco pose; depth: plane fit inside the board; hand-eye from cell.yaml"])
    res = [measure(c, board, dic, cfg, he) for c in caps]
    d = np.array([r["depth_minus_colour_mm"] for r in res])
    rec = os.path.join(cell.WS, "calib", time.strftime("depth_vs_colour_%Y%m%d_%H%M%S.json"))
    run_io.write_json(rec, dict(per_capture=res, mean_depth_minus_colour_mm=round(float(d.mean()), 2)))
    lines = [f"{'capture':40s} {'dist m':>7} {'depth-colour mm':>16} {'angle deg':>9} {'plane rms mm':>12} {'z colour':>9} {'z depth':>9}"]
    lines += [f"{r['capture'][-40:]:40s} {r['board_distance_m']:7.3f} {r['depth_minus_colour_mm']:16.2f} {r['plane_angle_deg']:9.3f} "
              f"{r['depth_plane_rms_mm']:12.3f} {r['board_z_base_colour_m']:9.4f} {r['board_z_base_depth_m']:9.4f}" for r in res]
    verdict = ("depth and colour agree: a height error is in the hand-eye -> re-calibrate" if abs(d.mean()) < 1.5 else
               "depth is offset from colour along the viewing axis: fix the depth alignment / scale, not the hand-eye")
    lines += [f"mean depth - colour along the viewing axis: {d.mean():+.2f} mm  -> {verdict}", f"-> {rec}"]
    run_io.show(NAME, "OUT", lines)


if __name__ == "__main__":
    main()
