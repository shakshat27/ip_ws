"""The cell's obstacles, ONE list for every planner (cell.yaml `scene:`, ROBOT BASE frame = our DH base, world-aligned).
Built from cell.usd by tools/export_scene_meshes.py. Each backend turns it into its own world model: cuRobo scene (cuboids +
meshes), MoveIt planning-scene collision objects, RoboDK station shapes. Meshes are read from .obj (triangles, metres)."""
import os
import numpy as np
import cell


def load(cfg):
    """-> dict(cuboids=[{name, centre, dims}], meshes=[{name, vertices Nx3, faces Mx3}]), enabled entries only."""
    sc = cfg.get("scene") or {}
    out = dict(cuboids=[dict(name=c["name"], centre=list(c["centre"]), dims=list(c["dims"])) for c in sc.get("cuboids", []) if c.get("enabled", True)],
               meshes=[])
    for m in sc.get("meshes", []):
        if not m.get("enabled", True):
            continue
        V, F = [], []
        for ln in open(os.path.join(cell.WS, m["file"])):
            t = ln.split()
            if t and t[0] == "v": V.append([float(x) for x in t[1:4]])
            elif t and t[0] == "f": F.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
        out["meshes"].append(dict(name=m["name"], vertices=np.array(V), faces=np.array(F, int)))
    return out


def box_triangles(centre, dims):
    """12 triangles (36 x 3 vertices) of an axis-aligned box."""
    c, h = np.asarray(centre, float), np.asarray(dims, float) / 2
    P = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([P[i] for q in quads for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])) for i in tri])
