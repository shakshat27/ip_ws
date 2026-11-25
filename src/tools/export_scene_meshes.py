#!/usr/bin/env python3
"""Export the cell's obstacle meshes from cell.usd into the robot BASE frame (world minus the robot origin; our DH base is
world-aligned) as .obj files in models/scene/ + models/scene/scene.json. ONE source for every planner (cuRobo, MoveIt, RoboDK).
Ignored on purpose (operator, 2026-11-24): every belt-looking object under /World/floor (Conveyor_stand, Rail, Base, Base_001,
Gear_*), pegs/bolts, charuco board, walls. The belt itself (/World/Conveyors/Plane) is exported as the surface the pick rests on.
  ~/isaac-sim-standalone-6.0.0-linux-x86_64/python.sh ~/ip_ws/src/tools/export_scene_meshes.py     (headless, Isaac closed)
"""
import json, os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from pxr import Usd, UsdGeom
OUT = os.path.expanduser("~/ip_ws/models/scene")
ITEMS = {"belt": "/World/Conveyors/Plane", "table_top": "/World/floor/Cube", "table_top_2": "/World/floor/Cube_001",
         "table_legs": "/World/floor/TableLegs", "table_legs_2": "/World/floor/TableLegs_001", "basket_a": "/World/floor/Basket",
         "basket_b": "/World/floor/Basket01", "robot_pedestal": "/World/floor/Robot_BAse"}
st = Usd.Stage.Open(os.path.expanduser("~/ip_ws/cell.usd"))
st.GetPrimAtPath("/World/Conveyors").SetActive(True)          # in memory only (the saved file now has it enabled)
r = UsdGeom.Xformable(st.GetPrimAtPath("/World/ur10")).ComputeLocalToWorldTransform(0).ExtractTranslation()
origin = np.array([r[0], r[1], r[2]])
meta = dict(origin_world=origin.tolist(), frame="robot base (world minus origin, axes world-aligned)", items={})
for name, path in ITEMS.items():
    root = st.GetPrimAtPath(path); V, F = [], []
    for q in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
        if not q.IsA(UsdGeom.Mesh): continue
        m = UsdGeom.Mesh(q); M = UsdGeom.Xformable(q).ComputeLocalToWorldTransform(0)
        pts = np.array([list(M.Transform(v)) for v in m.GetPointsAttr().Get()]) - origin
        cnt, idx, o = m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get(), 0
        for c in cnt:                                          # fan triangulation
            for k in range(1, c - 1): F.append((len(V) + idx[o], len(V) + idx[o + k], len(V) + idx[o + k + 1]))
            o += c
        V.extend(pts.tolist())
    V = np.array(V)
    with open(f"{OUT}/{name}.obj", "w") as f:
        f.writelines(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in V); f.writelines(f"f {a+1} {b+1} {c+1}\n" for a, b, c in F)
    meta["items"][name] = dict(prim=path, vertices=len(V), triangles=len(F), min=V.min(0).round(4).tolist(), max=V.max(0).round(4).tolist())
    print("X", name, len(V), len(F), V.min(0).round(3), V.max(0).round(3))
json.dump(meta, open(f"{OUT}/scene.json", "w"), indent=1)
app.close()
