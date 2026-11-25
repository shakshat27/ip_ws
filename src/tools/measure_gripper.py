#!/usr/bin/env python3
"""Measure the gripper's TCP (fingertip centre) and closing axis WITH THE WRIST CAMERA, in the flange (tool0) frame.

The wrist camera looks along the tool axis and sees the gripper at the top of its image. In depth:
  1 gripper pixels = everything closer than --near (default 0.30 m)
  2 fingertips     = the deepest --tip-band (default 12 mm) of the gripper: two blobs, one per finger
  3 each blob's 3D centre (camera frame) -> tool0 frame with the calibrated hand-eye (cell.yaml)
  4 TCP = midpoint of the two fingertip centres; closing axis = the direction between them (its yaw in tool0 X-Y)
Several captures are averaged (spread reported). Real cell: identical (a camera that sees its own gripper).
Bonus: boxes are measured through the same camera and hand-eye, so most of the hand-eye error cancels between the
box position and the fingertip position.
Measured with the gripper OPEN (the state in which it approaches a box).
  IN : capture folders (rgb.png, depth.npy, camera_info.json); hand-eye from cell.yaml
  OUT: ~/ip_ws/calib/gripper_<stamp>.json + overlay png; with --write: cell.yaml gripper.tool0_T_tcp,
       gripper.closing_axis_yaw_in_tool0 (backup <file>.<stamp>.bak next to it)

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/measure_gripper.py --fresh 3            # take 3 captures now (s2) and measure
  python3 ~/ip_ws/src/tools/measure_gripper.py --capture <run>/2_capture_x [...] [--write]
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import cell, run_io, kinematics as kin, geometry as geo

NAME = "measure gripper"
PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")


def measure(cap, he, near, band):
    d = np.load(os.path.join(cap, "depth.npy")); K = np.array(run_io.read_json(os.path.join(cap, "camera_info.json"))["K"])
    pts, valid = geo.backproject(d, K)
    grip = valid & (d < near)
    if grip.sum() < 1000:
        raise RuntimeError(f"{cap}: no gripper in view (only {int(grip.sum())} pixels nearer than {near} m)")
    zmax = pts[grip][:, 2].max()
    tips = grip & (pts[..., 2] > zmax - band)
    n, lab, st, _ = cv2.connectedComponentsWithStats(tips.astype(np.uint8))
    blobs = sorted([i for i in range(1, n) if st[i, 4] > 30], key=lambda i: -st[i, 4])[:2]
    if len(blobs) != 2:
        raise RuntimeError(f"{cap}: expected 2 fingertip blobs, found {len(blobs)}")
    cs = [pts[lab == i].mean(0) for i in blobs]
    cs = sorted(cs, key=lambda c: c[0])                                      # left finger first (image x)
    mid = (he @ np.append((cs[0] + cs[1]) / 2, 1))[:3]
    ax = he[:3, :3] @ (cs[1] - cs[0])
    yaw = math.atan2(ax[1], ax[0])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2                        # a closing axis has no sign: fold into -90..90 deg
    rgb = cv2.imread(os.path.join(cap, "rgb.png")); vis = rgb.copy()
    vis[grip] = (0.6 * vis[grip] + 0.4 * np.array([0, 140, 255])).astype(np.uint8); vis[tips] = (0, 0, 255)
    for c in cs:
        u, v = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]; cv2.circle(vis, (int(u), int(v)), 10, (255, 255, 0), 2)
    return dict(capture=cap, tcp_tool0=mid, closing_yaw=yaw, gap_m=float(np.linalg.norm(cs[1] - cs[0])),
                tilt_deg=math.degrees(math.asin(min(1.0, abs(ax[2]) / np.linalg.norm(ax)))), tips_px=[int(st[i, 4]) for i in blobs]), vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", nargs="*", default=[], help="capture folders to measure")
    ap.add_argument("--fresh", type=int, default=0, help="take this many captures now with s2 (arm must be still)")
    ap.add_argument("--near", type=float, default=0.30); ap.add_argument("--tip-band", type=float, default=0.012)
    ap.add_argument("--write", action="store_true", help="store the result in cell.yaml (backup kept)")
    a = ap.parse_args()
    cfg = cell.load(); he = kin.T_from_cfg(cell.require(cfg, "hand_eye", "tool0_T_camera"))
    caps = list(a.capture)
    if a.fresh:
        run = run_io.new_run("gripper"); rn = os.path.basename(run)
        for k in range(a.fresh):
            r = subprocess.run([sys.executable, os.path.join(PIPE, "s2_capture.py"), "--label", f"gripper{k}", "--run", rn])
            if r.returncode != 0:
                raise SystemExit("capture failed")
            caps.append(os.path.join(run, f"2_capture_gripper{k}")); time.sleep(0.3)
    if not caps:
        raise SystemExit("give --capture folders or --fresh N")
    run_io.show(NAME, "IN", [f"{len(caps)} capture(s): {[os.path.relpath(c, cell.RUNS) for c in caps]}",
                             f"gripper = nearer than {a.near} m; fingertips = deepest {a.tip_band * 1000:.0f} mm; hand-eye from cell.yaml"])
    res, vis = [], None
    for c in caps:
        r, v = measure(c, he, a.near, a.tip_band); res.append(r); vis = v
    T = np.array([r["tcp_tool0"] for r in res]); Y = np.array([r["closing_yaw"] for r in res])
    tcp, yaw = T.mean(0), float(np.mean(Y))
    stamp = time.strftime("%Y%m%d_%H%M%S"); os.makedirs(os.path.join(cell.WS, "calib"), exist_ok=True)
    rec = os.path.join(cell.WS, "calib", f"gripper_{stamp}.json")
    run_io.write_json(rec, dict(tcp_tool0_m=tcp.round(5).tolist(), tcp_spread_mm=(T.std(0) * 1000).round(2).tolist(),
                                closing_axis_yaw_in_tool0_rad=round(yaw, 6), closing_yaw_spread_deg=round(float(np.degrees(Y.std())), 3),
                                per_capture=[dict(capture=os.path.relpath(r["capture"], cell.RUNS), tcp_tool0_m=np.round(r["tcp_tool0"], 5).tolist(),
                                                  closing_yaw_deg=round(math.degrees(r["closing_yaw"]), 3), gap_mm=round(r["gap_m"] * 1000, 1),
                                                  tilt_deg=round(r["tilt_deg"], 2), tip_pixels=r["tips_px"]) for r in res],
                                method="wrist camera: deepest gripper pixels = fingertips; midpoint = TCP; hand-eye from cell.yaml; gripper open"))
    cv2.imwrite(rec.replace(".json", ".png"), vis)
    lines = [f"TCP in tool0: x {tcp[0] * 1000:+.1f}  y {tcp[1] * 1000:+.1f}  z {tcp[2] * 1000:+.1f} mm  (spread {np.round(T.std(0) * 1000, 2).tolist()} mm)",
             f"closing axis yaw in tool0: {math.degrees(yaw):+.2f} deg (spread {np.degrees(Y.std()):.3f} deg); "
             f"fingertip gap {np.mean([r['gap_m'] for r in res]) * 1000:.1f} mm (open); axis tilt {np.mean([r['tilt_deg'] for r in res]):.2f} deg",
             f"-> {rec} (+ overlay .png)"]
    if a.write:
        path = cell.CONFIG if hasattr(cell, "CONFIG") else os.path.join(cell.WS, "config", "cell.yaml")
        bak = path + time.strftime(".%Y%m%d-%H%M%S.bak"); shutil.copy2(path, bak)
        txt = open(path).read()
        old_tcp = next(l for l in txt.splitlines() if l.strip().startswith("tool0_T_tcp:"))
        old_ax = next(l for l in txt.splitlines() if l.strip().startswith("closing_axis_yaw_in_tool0:"))
        ind = old_tcp[: len(old_tcp) - len(old_tcp.lstrip())]
        txt = txt.replace(old_tcp, f"{ind}tool0_T_tcp:             # MEASURED {time.strftime('%Y-%m-%d %H:%M')} by the wrist camera ({os.path.basename(rec)}), gripper open\n"
                                   f"{ind}  xyz: [{tcp[0]:.5f}, {tcp[1]:.5f}, {tcp[2]:.5f}]\n{ind}  rpy: [0.0, 0.0, 0.0]")
        txt = txt.replace(old_ax, f"{ind}closing_axis_yaw_in_tool0: {yaw:.6f}   # rad, MEASURED {time.strftime('%Y-%m-%d %H:%M')} by the wrist camera")
        open(path, "w").write(txt); lines.append(f"written to cell.yaml (backup {bak})")
    run_io.show(NAME, "OUT", lines)


if __name__ == "__main__":
    main()
