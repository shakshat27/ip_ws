#!/usr/bin/env python3
"""Script 4b - GRASP. Turns a chosen object into gripper goal poses (rule-based look-then-grasp).

  refine_look : the flange pose that puts the CAMERA straight above the object, looking down, at
                refine_capture.standoff - for a second, closer capture that fixes the yaw precisely
  grasp       : fingertip centre at the object centre, depth = grip_depth_fraction of the height,
                never closer than surface_clearance to the surface; fingers close across a pair of
                faces, using the face direction nearest the current wrist so the wrist turns least
  pre_grasp   : grasp + approach_height straight up (also the lift point)
  opening     : object width + margin (refused above the gripper's maximum)
All poses are tool0 (flange) poses in the robot base frame; script 5 turns them into motion.
  IN : <run>/4_localize_<label>/objects.json, the capture's joints (current wrist), cell.yaml
  OUT: <run>/4b_grasp_<label>/grasp.json

  python3 ~/ip_ws/src/pipeline/s4b_grasp.py [--run latest] [--label pick_overview] [--object 0]
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell, run_io, geometry as geo, kinematics as kin

NAME = "s4b grasp"
DOWN = kin.rpy_to_R(math.pi, 0.0, 0.0)          # tool or camera Z pointing straight down


def pose_out(T):
    return dict(xyz=[round(float(v), 5) for v in T[:3, 3]], rpy=[round(float(v), 6) for v in kin.R_to_rpy(T[:3, :3])])


def finger_clearance(centre, jaw_deg, opening, objs, self_id, pad, surf=None):
    """Smallest distance (m, negative = overlap) from either finger pad's footprint, at the approach opening, to any other
    object's footprint rectangle and, with `surf`, to the edge of the surface the object rests on (beyond its edge are the
    conveyor's frame and end stop). Pads are sampled as points across their thickness and width."""
    u = np.array([math.cos(math.radians(jaw_deg)), math.sin(math.radians(jaw_deg))]); v = np.array([-u[1], u[0]])
    pts = [np.asarray(centre) + s * u * (opening / 2 + t) + v * w
           for s in (1, -1) for t in np.linspace(0, pad["thickness"], 4) for w in np.linspace(-pad["width"] / 2, pad["width"] / 2, 9)]
    best = math.inf
    if surf is not None:
        a = surf["yaw"]; Rs = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]); h = np.array(surf["size"]) / 2
        for p in pts:
            best = min(best, float(np.min(h - np.abs(Rs @ (p - np.array(surf["centre"][:2]))))))
    for n in objs:
        if n["id"] == self_id:
            continue
        a = math.radians(n["yaw_deg"]); Rn = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        h = np.array(n["footprint"]) / 2
        for p in pts:
            d = np.abs(Rn @ (p - np.array(n["position"][:2]))) - h
            best = min(best, float(np.linalg.norm(np.maximum(d, 0)) + min(max(d[0], d[1]), 0)))
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="latest"); ap.add_argument("--label")
    ap.add_argument("--object", type=int, help="object id from objects.json; default: the first")
    a = ap.parse_args()
    cfg = cell.load(); he = kin.T_from_cfg(cell.require(cfg, "hand_eye", "tool0_T_camera"))
    run = run_io.resolve(a.run)
    loc = os.path.join(run, f"4_localize_{a.label}") if a.label else run_io.latest_step(run, "4_localize_")
    label = os.path.basename(loc).replace("4_localize_", "")
    objs = run_io.read_json(os.path.join(loc, "objects.json"))["objects"]
    if not objs:
        raise SystemExit("no objects to grasp")
    o = next(x for x in objs if x["id"] == a.object) if a.object is not None else objs[0]
    det = run_io.read_json(os.path.join(run, f"3_detect_{label}", "detections.json"))
    joints = run_io.read_json(os.path.join(run, det["capture"], "capture.json"))["joints"]
    T_now = kin.fk(joints, cfg["robot"]["dh"])
    run_io.show(NAME, "IN", [f"{loc}/objects.json: object #{o['id']} {o['type']} at {np.round(o['position'], 4).tolist()} m, yaw {o['yaw_deg']:.1f} deg",
                             f"current wrist yaw {geo.yaw_deg(T_now[:3, :3]):.1f} deg (from the capture's joints)",
                             "standoff, gripper, grasp rules, hand-eye from cell.yaml"])
    x, y, _ = o["position"]; notes = []

    # refine look: camera straight down above the object, keeping the camera's current heading
    cam_now = T_now @ he
    cam_yaw = math.atan2(cam_now[1, 0], cam_now[0, 0])
    Tc = np.eye(4); Tc[:3, :3] = kin.rpy_to_R(0, 0, cam_yaw) @ DOWN
    Tc[:3, 3] = [x, y, o["top_z"] + cfg["refine_capture"]["standoff"]]
    refine = Tc @ np.linalg.inv(he)

    # finger closing direction: across a pair of faces that fits the gripper and leaves the fingers clear of every other
    # object; among those, the one nearest the current wrist heading (footprint[0] lies along yaw_deg, as cv2.minAreaRect)
    # If a pad crosses a wall or a neighbour, the grip point may slide along the finger width (off-centre on the box face),
    # as long as the pad centre stays 5 mm inside the face.
    face = o["yaw_deg"]; wrist = geo.yaw_deg(T_now[:3, :3]); gc = cfg["gripper"]; need = cfg["grasp"]["finger_clearance"]
    surf = cfg["surfaces"][o["surface"]]
    cands = []
    for k in range(-4, 5):
        c = face + k * 90
        w, side = o["footprint"][k % 2], o["footprint"][(k + 1) % 2]
        op = w + gc["opening_margin"]
        v = np.array([-math.sin(math.radians(c)), math.cos(math.radians(c))])
        smax = max(0.0, side / 2 - 0.005)
        tries = []
        for s in sorted(np.arange(-smax, smax + 1e-4, 0.001), key=abs):
            ctr = np.array([x, y]) + s * v
            tries.append((finger_clearance(ctr, c, op, objs, o["id"], gc["pad"], surf), s, ctr))
            if tries[-1][0] >= need:
                break
        clr, s, ctr = tries[-1] if tries[-1][0] >= need else max(tries, key=lambda t: t[0])
        cands.append(dict(jaw=c, width=w, opening=op, fits=op <= gc["max_opening"], clearance=clr, shift=float(s), centre=ctr,
                          turn=abs(geo.wrap(c - wrist, 360))))
    ok = [c for c in cands if c["fits"] and c["clearance"] >= need]
    if ok:
        best = min(ok, key=lambda c: c["turn"])
    else:
        best = max([c for c in cands if c["fits"]] or cands, key=lambda c: (c["fits"], c["clearance"]))
        notes.append(f"NO SAFE FACE PAIR: best finger clearance {best['clearance']*1000:+.1f} mm (needs {need*1000:.0f} mm)"
                     + ("" if best["fits"] else f"; object {best['width']*1000:.0f} mm too wide for the gripper"))
    jaw, width, opening = best["jaw"], best["width"], best["opening"]
    gx, gy = best["centre"]
    # the TCP is the pad centre (gripper.pad, /tf model): its lower end, half a pad length below, must clear the surface
    z_tcp = max(o["top_z"] - cfg["grasp"]["grip_depth_fraction"] * o["height"],
                cfg["surfaces"][o["surface"]]["centre"][2] + cfg["grasp"]["surface_clearance"] + gc["pad"]["length"] / 2)

    out = dict(object=o, refine_look=pose_out(refine), jaw_direction_deg=round(geo.wrap(jaw, 360), 2),
               opening_m=round(opening, 4), tcp_target=[round(float(gx), 5), round(float(gy), 5), round(z_tcp, 5)],
               finger_clearance_m=None if math.isinf(best["clearance"]) else round(best["clearance"], 4),
               grip_offset_m=round(best["shift"], 4))
    tab = cfg["gripper"].get("opening_table")
    if tab:                                        # finger angle for the approach opening; TCP depth when the fingers touch
        g, q, zt = (np.array(tab[k]) for k in ("gap_m", "finger_joint_rad", "tcp_z_m")); i = np.argsort(g)
        out["finger_approach_rad"] = round(float(np.interp(opening, g[i], q[i])), 4)
        out["finger_close_rad"] = cfg["gripper"].get("close_rad", float(q.max()))
        out["finger_contact_rad"] = round(float(np.interp(width, g[i], q[i])), 4)   # where the fingers should stop on this object
        out["tcp_z_at_contact_m"] = round(float(np.interp(width, g[i], zt[i])), 5)
    tcp, axis = cfg["gripper"]["tool0_T_tcp"], cfg["gripper"]["closing_axis_yaw_in_tool0"]
    if tcp is None or axis is None:
        notes.append("grasp / pre-grasp flange poses pending: gripper.tool0_T_tcp and closing_axis_yaw_in_tool0 not measured yet")
    else:
        T_tcp = kin.T_from_cfg(tcp)
        if "tcp_z_at_contact_m" in out:
            T_tcp[2, 3] = out["tcp_z_at_contact_m"]
        Tt = np.eye(4); Tt[:3, :3] = kin.rpy_to_R(0, 0, math.radians(jaw) - axis) @ DOWN; Tt[:3, 3] = out["tcp_target"]
        grasp = Tt @ np.linalg.inv(T_tcp)
        pre = grasp.copy(); pre[2, 3] += cfg["grasp"]["approach_height"]
        out.update(grasp=pose_out(grasp), pre_grasp=pose_out(pre), lift=pose_out(pre))
    out["notes"] = notes
    dst = run_io.step_dir(run, f"4b_grasp_{label}")
    run_io.write_json(os.path.join(dst, "grasp.json"), out)
    lines = [f"{dst}/grasp.json",
             f"refine_look flange pose: xyz {out['refine_look']['xyz']} m, rpy {out['refine_look']['rpy']} rad "
             f"(camera {cfg['refine_capture']['standoff']} m above the object, looking down)",
             f"fingers close along {out['jaw_direction_deg']} deg, opening {opening*1000:.0f} mm, TCP (pad centre) at {out['tcp_target']} m",
             "finger clearance to other objects and the surface edge: "
             + ("none nearby" if out["finger_clearance_m"] is None else f"{out['finger_clearance_m'] * 1000:+.1f} mm")
             + f"; grip point {out['grip_offset_m'] * 1000:+.0f} mm off the face centre"]
    if "finger_approach_rad" in out:
        lines.append(f"finger_joint: approach {out['finger_approach_rad']} rad, close {out['finger_close_rad']} rad; "
                     f"TCP {out['tcp_z_at_contact_m'] * 1000:.1f} mm below the flange at contact")
    if "grasp" in out:
        lines += [f"pre_grasp {out['pre_grasp']}", f"grasp     {out['grasp']}"]
    run_io.show(NAME, "OUT", lines + [f"note: {n}" for n in notes])


if __name__ == "__main__":
    main()
