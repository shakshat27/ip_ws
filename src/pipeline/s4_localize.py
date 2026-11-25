#!/usr/bin/env python3
"""Script 4 - LOCALIZE. Turns each detection into an object type and a pose in the robot base frame.

For each mask: its depth pixels -> 3D points in the base frame (FK x hand-eye) -> the top face
(highest 5 mm) -> smallest rectangle around it: centre, footprint, yaw; height above the surface.
Type: the detector's class if it has one; otherwise the catalogue object (cell.yaml objects)
whose size matches the measured size.
  IN : <run>/2_capture_<label>/, <run>/3_detect_<label>/detections.json + masks, cell.yaml
  OUT: <run>/4_localize_<label>/objects.json (per object: type, position, yaw, size, surface)

  python3 ~/ip_ws/src/pipeline/s4_localize.py [--run latest] [--label pick_overview]
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import cell, run_io, geometry as geo

NAME = "s4 localize"


def measure(P, surface_z):
    top_z = np.percentile(P[:, 2], 95)
    top = P[P[:, 2] > top_z - 0.005]
    (mx, my), (a, b), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
    height = float(top_z - surface_z)
    return dict(position=[mx / 1000, my / 1000, surface_z + height / 2], yaw_deg=float(ang),
                footprint=[a / 1000, b / 1000], height=height, top_z=float(top_z))


def match_type(m, catalogue, tol=0.3):
    """Catalogue object whose (sorted footprint, height) is closest, within tol (relative) on every dimension."""
    best = None
    meas = sorted(m["footprint"]) + [m["height"]]
    for name, o in catalogue.items():
        size = o["size"]; ref = sorted(size[:2]) + [size[2]]
        rel = max(abs(x - r) / r for x, r in zip(meas, ref))
        if rel <= tol and (best is None or rel < best[1]):
            best = (name, rel)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="latest"); ap.add_argument("--label")
    a = ap.parse_args()
    cfg = cell.load(); cell.require(cfg, "hand_eye", "tool0_T_camera")
    run = run_io.resolve(a.run)
    det_dir = os.path.join(run, f"3_detect_{a.label}") if a.label else run_io.latest_step(run, "3_detect_")
    label = os.path.basename(det_dir).replace("3_detect_", "")
    dets = run_io.read_json(os.path.join(det_dir, "detections.json"))
    cap = os.path.join(run, dets["capture"])
    depth = np.load(os.path.join(cap, "depth.npy")); K = run_io.read_json(os.path.join(cap, "camera_info.json"))["K"]
    joints = run_io.read_json(os.path.join(cap, "capture.json"))["joints"]
    run_io.show(NAME, "IN", [f"{det_dir}/detections.json: {len(dets['detections'])} detection(s) ({dets['method']})",
                             f"{cap}/ depth + joints {np.round(joints, 4).tolist()}", "hand-eye, surfaces, object catalogue from cell.yaml"])
    pts, valid = geo.backproject(depth, K, near=cfg["detection"]["near_cutoff"])
    B = geo.to_frame(pts, geo.base_T_camera(joints, cfg))
    objects = []
    for d in dets["detections"]:
        mask = cv2.imread(os.path.join(det_dir, d["mask"]), 0) > 0
        P = B[mask & valid]
        surf_z = cfg["surfaces"][d["surface"]]["centre"][2]
        m = measure(P, surf_z)
        if d["cls"] != "unknown":
            otype, how = d["cls"], "detector class"
        else:
            hit = match_type(m, cfg["objects"])
            otype, how = (hit[0], f"size match, worst dimension off {hit[1]*100:.0f}%") if hit else ("unknown", "no catalogue size match")
        sym = cfg["objects"].get(otype, {}).get("yaw_symmetry_deg", 360)
        objects.append(dict(id=d["id"], type=otype, type_source=how, surface=d["surface"], position=m["position"],
                            yaw_deg=m["yaw_deg"], yaw_symmetry_deg=sym, footprint=m["footprint"], height=m["height"],
                            top_z=m["top_z"], points=int(len(P)), frame="robot base"))
    out = run_io.step_dir(run, f"4_localize_{label}")
    run_io.write_json(os.path.join(out, "objects.json"), dict(detections=os.path.basename(det_dir), objects=objects))
    run_io.show(NAME, "OUT", [f"{out}/objects.json: {len(objects)} object(s), robot base frame"] +
                [f"#{o['id']} {o['type']} ({o['type_source']}): position {np.round(o['position'], 4).tolist()} m, "
                 f"yaw {o['yaw_deg']:.1f} deg (repeats every {o['yaw_symmetry_deg']}), "
                 f"footprint {o['footprint'][0]*1000:.0f}x{o['footprint'][1]*1000:.0f} mm, height {o['height']*1000:.0f} mm" for o in objects])


if __name__ == "__main__":
    main()
