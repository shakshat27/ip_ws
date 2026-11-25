#!/usr/bin/env python3
"""Build cuRobo's robot config for OUR arm (UR10 CB3 + Robotiq 2F-85 + wrist D455) from the manufacturers' data.

cuRobo ships UR10e models only (different dimensions). Here:
  kinematics : the UR10 URDF generated from UR's ur_description (xacro ur_type:=ur10; identical to our DH, 4e-10 checked)
  arm spheres: fitted by cuRobo's own fitter to UR's UR10 collision meshes, placed with each link's collision origin
  tool spheres (on tool0): Robotiq 2F-85 body and open fingers from its datasheet size, D455 from its datasheet size at the
               mount measured by the hand-eye calibration (cell.yaml hand_eye.tool0_T_camera)
Run in the cuRobo environment.
  IN : ~/ip_ws/models/curobo_ur10/ur10.urdf, ur_description meshes, cell.yaml
  OUT: ~/ip_ws/models/curobo_ur10/ur10_cell.yml (+ a printed sphere count per link)

  ~/ip_ws/.venv_curobo/bin/python ~/ip_ws/src/tools/make_curobo_ur10.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, trimesh, yaml, yourdfpy
import cell, kinematics as kin
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

OUT = os.path.join(cell.WS, "models", "curobo_ur10")
URDF = os.path.join(OUT, "ur10.urdf")
ARM = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
SPHERES = {"shoulder_link": 4, "upper_arm_link": 10, "forearm_link": 9, "wrist_1_link": 3, "wrist_2_link": 3, "wrist_3_link": 2}


def mesh_path(fn):
    if fn.startswith("package://"):
        pkg, rest = fn[len("package://"):].split("/", 1)
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory(pkg), rest)
    return fn


def tool_spheres(cfg):
    """Gripper + camera on tool0 (z along the tool axis). 2F-85 (datasheet): body ~ 0.075 x 0.087 m, coupling + body to
    the finger base ~ 0.11 m; open fingers reach +-0.06 m along the closing axis down to the pads (TCP 0.129 m). D455:
    124 x 26 x 29 mm."""
    s = []
    for z in (0.02, 0.06, 0.095):                                   # coupling + gripper body
        s.append(([0.0, 0.0, z], 0.05))
    ax = cfg["gripper"]["closing_axis_yaw_in_tool0"] or 0.0          # closing axis direction in tool0 x-y
    u = np.array([np.cos(ax), np.sin(ax), 0.0])
    for side in (1, -1):                                            # open fingers: outer links + pads
        for r, z, rad in ((0.055, 0.115, 0.022), (0.05, 0.14, 0.018)):
            s.append(((side * r * u + [0, 0, z]).tolist(), rad))
    he = kin.T_from_cfg(cfg["hand_eye"]["tool0_T_camera"])          # camera body: spheres along its long (x) axis
    for dx in (-0.045, 0.0, 0.045):
        s.append(((he @ np.array([dx, 0.0, 0.0, 1.0]))[:3].tolist(), 0.022))
    return [dict(center=[round(float(v), 4) for v in c], radius=r) for c, r in s]


def main():
    cfg = cell.load()
    u = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=False)
    spheres = {}
    for link in ARM:
        col = u.link_map[link].collisions[0]
        m = trimesh.load(mesh_path(col.geometry.mesh.filename), force="mesh")
        if col.geometry.mesh.scale is not None:
            m.apply_scale(col.geometry.mesh.scale)
        if col.origin is not None:
            m.apply_transform(col.origin)                           # collision origin -> link frame
        res = fit_spheres_to_mesh(m, num_spheres=SPHERES[link], fit_type=SphereFitType.MORPHIT, compute_metrics=True)
        c, r = np.asarray(res.centers.cpu() if hasattr(res.centers, "cpu") else res.centers), np.asarray(res.radii.cpu() if hasattr(res.radii, "cpu") else res.radii)
        spheres[link] = [dict(center=[round(float(v), 4) for v in cc], radius=round(float(rr), 4)) for cc, rr in zip(c, r)]
        met = res.metrics
        print(f"{link:16s} {len(spheres[link]):2d} spheres  mesh extent {np.round(m.extents, 3).tolist()} m"
              + (f"  coverage {getattr(met, 'coverage', float('nan')):.2f}" if met is not None else ""), flush=True)
    spheres["tool0"] = tool_spheres(cfg)
    print(f"{'tool0':16s} {len(spheres['tool0']):2d} spheres  (2F-85 body + open fingers + D455)")
    robot = {"robot_cfg": {"kinematics": {
        "urdf_path": URDF, "asset_root_path": OUT, "base_link": "base_link", "tool_frames": ["tool0"],
        "collision_link_names": ARM + ["tool0"], "collision_spheres": spheres, "collision_sphere_buffer": 0.005,
        "self_collision_buffer": {k: 0.0 for k in ARM + ["tool0"]},
        "self_collision_ignore": {"shoulder_link": ["upper_arm_link"], "upper_arm_link": ["forearm_link"],
                                  "forearm_link": ["wrist_1_link"], "wrist_1_link": ["wrist_2_link", "wrist_3_link"],
                                  "wrist_2_link": ["wrist_3_link", "tool0"], "wrist_3_link": ["tool0"]},
        "mesh_link_names": [], "lock_joints": None,
        "cspace": {"joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                   "default_joint_position": cfg["capture_poses"]["pick_overview"]["joints"],
                   "cspace_distance_weight": [1.0] * 6, "null_space_weight": [1.0] * 6,
                   "max_acceleration": 12.0, "max_jerk": 500.0, "position_limit_clip": 0.0}}}}
    path = os.path.join(OUT, "ur10_cell.yml")
    yaml.safe_dump(robot, open(path, "w"), sort_keys=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
