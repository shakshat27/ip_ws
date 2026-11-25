#!/usr/bin/env python3
"""Offline test of scripts 3 and 4 (no Isaac, no ROS): render a synthetic depth image of a box on
a belt from a chosen camera pose by ray casting, run the real detection and localization code,
and check the recovered position, yaw and type. Several camera poses, including tilted ones.

  python3 ~/ip_ws/src/tests/test_detect_localize.py
"""
import copy, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), os.path.join(HERE, "..", "pipeline")]
import numpy as np
import cell, geometry as geo, kinematics as kin
import s3_detect, s4_localize

W, H, F, CX, CY = 1280, 800, 634.1, 640.0, 400.0
K = [[F, 0, CX], [0, F, CY], [0, 0, 1]]
SIDE = 0.035
CFG = copy.deepcopy(cell.load())
CFG["hand_eye"]["tool0_T_camera"] = CFG["hand_eye"]["tool0_T_camera"] or {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}


def cam_pose(pos, yaw_deg, tilt_deg):
    """Optical-frame camera pose looking down, turned by yaw, tilted about its x axis."""
    T = np.eye(4); T[:3, :3] = kin.rpy_to_R(0, 0, math.radians(yaw_deg)) @ kin.rpy_to_R(math.pi + math.radians(tilt_deg), 0, 0)
    T[:3, 3] = pos; return T


def render(Tc, belt, box_xy, box_yaw):
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u)], -1) @ Tc[:3, :3].T   # optical Z = 1 -> hit distance = depth
    o = Tc[:3, 3]; best = np.full((H, W), np.inf); bz = belt["centre"][2]
    for zp, rect in ((bz, belt), (bz - 0.46, None)):
        s = (zp - o[2]) / d[..., 2]; p = o + d * s[..., None]; ok = s > 0
        if rect is not None:
            ok &= geo.in_rectangle(p, rect)
        best = np.where(ok & (s < best), s, best)
    c, sn = math.cos(math.radians(box_yaw)), math.sin(math.radians(box_yaw))
    Rb = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]); centre = np.array([box_xy[0], box_xy[1], bz + SIDE / 2])
    ol, dl = (o - centre) @ Rb, d @ Rb
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-SIDE / 2 - ol) / dl, (SIDE / 2 - ol) / dl
    tmin, tmax = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
    hit = (tmax >= tmin) & (tmin > 0)
    return np.where(hit & (tmin < best), tmin, best).astype(np.float32)


CASES = [  # name, belt yaw (rad), camera position, camera yaw, tilt, box xy, box yaw
    ("straight down", 0.0, (0.20, -0.55, 0.40), 0, 0, (0.20, -0.53), 0),
    ("camera turned 90 deg", 0.0, (0.20, -0.55, 0.40), 90, 0, (0.24, -0.58), 30),
    ("camera tilted 20 deg", 0.0, (0.20, -0.40, 0.45), 0, -20, (0.18, -0.56), -40),   # tilt towards -y, where the box is
    ("belt turned 180 deg", math.pi, (0.20, -0.55, 0.40), 45, 10, (0.15, -0.52), 25),
]
fails = 0
for name, byaw, cpos, cyaw, tilt, bxy, byw in CASES:
    belt = {"centre": [0.2, -0.55, -0.0674], "size": [0.9, 0.42], "yaw": byaw}
    cfg = copy.deepcopy(CFG); cfg["surfaces"] = {"belt_a": belt}
    Tc = cam_pose(cpos, cyaw, tilt)
    depth = render(Tc, belt, bxy, byw)
    s3_detect.geo.base_T_camera = s4_localize.geo.base_T_camera = lambda joints, c: Tc   # the camera pose FK x hand-eye would give
    found = s3_detect.detect_geometry(depth, K, None, cfg)
    if len(found) != 1:
        print(f"FAIL {name}: expected 1 detection, got {len(found)}"); fails += 1; continue
    pts, valid = geo.backproject(depth, K, near=cfg["detection"]["near_cutoff"])
    m = s4_localize.measure(geo.to_frame(pts, Tc)[found[0]["mask"] & valid], belt["centre"][2])
    hit = s4_localize.match_type(m, cfg["objects"])
    e = (np.array(m["position"]) - np.array([bxy[0], bxy[1], belt["centre"][2] + SIDE / 2])) * 1000
    ey = geo.wrap(m["yaw_deg"] - byw, 90)
    ok = np.linalg.norm(e) < 1.5 and abs(ey) < 1.5 and hit is not None and hit[0] == "box"
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:22s} error {np.round(e, 2).tolist()} mm, yaw {ey:+.2f} deg, "
          f"type {hit[0] if hit else 'unknown'}, {found[0]['pixels']} px")
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
