"""Synthetic training data for the detector (CV-3), rendered by Isaac Replicator from the pick capture pose.

Run with Isaac's own Python (alias isaacpy), headless; runs beside the Isaac GUI (~2 GB GPU):
  isaacpy ~/ip_ws/src/tools/gen_dataset.py --out ~/ip_ws/datasets/box_v2 --split train --frames 2000 --seed 1
  ... --split valid --frames 300 --seed 2
  ... --split test  --frames 300 --seed 3 --palette heldout      # colours the model never trains on
  smoke test first:  ... --split smoke --frames 10 --seed 0

What it does (the scene file on disk is never modified):
  - opens ~/ip_ws/cell.usd in memory; deactivates the ROS action graphs (no topics from this process,
    which also runs on its own ROS domain) and the calibration board; hides the scene's own test cube
  - poses the robot at capture pose pick_overview (cell.yaml) through physics, so the gripper and the
    wrist camera are where they are on a real capture; every episode jitters the arm joints by <= +-1 deg
  - each frame: 0-5 boxes in the belt-end pick zone (random yaw, catalogue size +-10 %, random colour
    family), 0-2 unlabelled distractor shapes, random light intensity / tint
  - LABELS ARE COMPUTED BY US, not by Replicator's semantic labelling (which dropped every label in this
    scene, 2026-10-12): each depth pixel -> 3D point (camera pose from the robot's joint angles x UR10
    kinematics x the rigid wrist-to-camera offset; lens from the camera's own settings); a pixel belongs to
    box i if its point lies inside box i's known 3D outline. Occlusion is automatic.
  - a fixed set of boxes/distractors, rearranged per layout with the timeline STOPPED (objects changed during play
    render wrongly in colour; creating objects per frame exhausts the renderer: see 'objects' below); then play and
    pose the arm again. ~0.65 frames/s
  - box colours: palette values are screen (sRGB) colours, converted to linear for USD displayColor
  - writes <out>/<split>/  NNNNNN.jpg + _annotations.coco.json (boxes AND polygon masks, the format
    RF-DETR trains on) and, per frame, NNNNNN_depth.npy (float16, metres) + NNNNNN_gt.json (the arm's joint
    angles for that image, and the true box poses in the robot base frame: for scoring only)
Simulator knowledge (belt geometry, object poses) is used here only to BUILD scenes and LABELS; the
runtime pipeline never sees it.
"""
import argparse, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/ip_ws/datasets/box_v2"))
ap.add_argument("--split", default="smoke")
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--frames-per-episode", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-new-frames", type=int, default=0, help="exit (code 3) after this many new frames, labels saved; 0 = no limit. "
                "Each frame's stop/play leaks renderer resources (Isaac 6.0): run_gen.sh restarts in chunks with --resume")
ap.add_argument("--resume", action="store_true", help="continue an interrupted split (same seed): frames in the saved labels file are kept, "
                "the next ones are identical to an uninterrupted run")
ap.add_argument("--palette", choices=["train", "heldout"], default="train")
ap.add_argument("--max-boxes", type=int, default=5)
ap.add_argument("--size-jitter", type=float, default=0.10)
ap.add_argument("--min-pixels", type=int, default=40, help="instances with fewer visible pixels are not labelled")
ap.add_argument("--stage", default=os.path.expanduser("~/ip_ws/cell.usd"))
ap.add_argument("--settle-updates", type=int, default=10, help="full app updates after each rearrangement, before capture")
ap.add_argument("--settle-frames", type=int, default=0, help="extra Replicator frames rendered and discarded")
ap.add_argument("--rt-subframes", type=int, default=32, help="render sub-frames per image; low values leave 'ghosts' of moved objects")
args = ap.parse_args()
os.environ["ROS_DOMAIN_ID"] = "99"                      # never on the live session's domain

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np, cv2, yaml
import omni.usd, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf
from isaacsim.core.experimental.prims import Articulation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import cell, kinematics as kin

CFG = cell.load(); rng = np.random.default_rng(args.seed)
CAM = "/World/ur10/wrist_3_link/rsd455/RSD455/Camera_OmniVision_OV9782_Color"
W, H = 1280, 800
PALETTES = {   # colour families (RGB 0-1); the held-out test palette shares none with training
    "train": [("cardboard", (0.55, 0.40, 0.22), 0.08), ("grey", (0.45, 0.45, 0.45), 0.20), ("green", (0.15, 0.65, 0.20), 0.08),
              ("blue", (0.15, 0.30, 0.70), 0.08), ("white", (0.85, 0.85, 0.82), 0.06)],
    "heldout": [("red", (0.75, 0.12, 0.10), 0.08), ("yellow", (0.85, 0.75, 0.15), 0.06), ("black", (0.06, 0.06, 0.07), 0.03),
                ("kraft", (0.68, 0.52, 0.33), 0.05)],
}

# ---------- scene ----------
DISTRACTOR_KINDS = ["sphere", "cylinder_lying", "sphere"]


ctx = omni.usd.get_context(); ctx.open_stage(args.stage)
for _ in range(20):
    app.update()
stage = ctx.get_stage()
for p in stage.Traverse():                               # no ROS from this process; no calibration board; no test cube
    ps = str(p.GetPath())
    if p.GetTypeName() == "OmniGraph" or ps.startswith("/Graphs"):
        p.SetActive(False)
    elif any(a.assetPath.endswith("charuco_board.usda") for a in p.GetPrimStack()[0].referenceList.GetAddedOrExplicitItems()) if p.GetPrimStack() else False:
        p.SetActive(False)
cube = stage.GetPrimAtPath("/World/Cube")
if cube:
    cube.SetActive(False)            # the scene's own test cube is not part of the dataset

# belt top and pick zone, from the scene geometry (scene building only)
belt = stage.GetPrimAtPath("/World/Conveyors/Plane")
M = UsdGeom.Xformable(belt).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
pts = np.array([M.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(belt).GetPointsAttr().Get()])
bx0, bx1, by0, by1, bz = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), float(pts[:, 2].mean())
# Boxes rest on the ROLLER TOPS, which stand above the belt plane (found 2026-10-12: rollers 7 mm above it).
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
rest_z = bz
for p in stage.Traverse():
    if p.GetTypeName() != "Mesh" or str(p.GetPath()).startswith(("/World/ur10", "/World/Conveyors", "/World/SDG")):
        continue
    b = cache.ComputeWorldBound(p).ComputeAlignedRange()
    if b.IsEmpty():
        continue
    mn, mx = b.GetMin(), b.GetMax()
    if mx[0] > bx0 and mn[0] < bx1 and mx[1] > by0 and mn[1] < by1 and bz < mx[2] < bz + 0.02 and (mx[0] - mn[0]) > 0.5:
        rest_z = max(rest_z, mx[2])       # long meshes just above the belt plane: the rollers
ZONE = dict(x=(bx1 - 0.42, bx1 - 0.04), y=(by0 + 0.05, by1 - 0.05), z=rest_z)  # belt end, inside the side rails
print(f"[gen] belt plane z {bz:.4f}, objects rest on z {rest_z:.4f} (roller tops), pick zone x {ZONE['x'][0]:.3f}..{ZONE['x'][1]:.3f}, y {ZONE['y'][0]:.3f}..{ZONE['y'][1]:.3f} (world)", flush=True)

base_T_world = np.linalg.inv(kin.T_from_cfg(CFG["referee"]["world_T_base"]))
box_size = np.array(CFG["objects"]["box"]["size"], float)


# ---------- objects: a fixed set, changed ONLY while the timeline is stopped ----------
# Found 2026-10-19 (operator's observation that pre-existing cubes render fine, then tests on box_v1 test frames 9, 24,
# 32, 34): objects moved OR created while the timeline plays (physics running) render wrongly in COLOUR (see-through
# over the rollers, invisible, or left behind as a ghost) while depth stays right. The same boxes at the same poses
# render correctly when they are put there before the timeline plays. Waiting, more sub-frames, FXAA instead of DLSS,
# lights, the rollers' displacement map, and a pre-built object pool made no difference. So each layout: stop the
# timeline, rearrange, play, pose the arm again.
# What actually cures it is the stop -> play transition (the renderer rebuilds its copy of the scene): pausing, or
# keeping the timeline stopped for the whole capture with the robot pose written in, did NOT (2026-10-19 tests).
# That rebuild leaks renderer resources: "[Fatal] [omni.rtx] Out of resource descriptors!" after 161-618 frames,
# then a HANG. Hence chunks: --max-new-frames + --resume, driven by run_gen.sh (fresh process per chunk).
# A FIXED set of objects, moved (never created per frame).
# Colour through displayColor with NO material (2026-10-12: a bound UsdPreviewSurface rendered see-through, OmniPBR
# created headless rendered not at all; RTX shades material-less geometry with displayColor).
SDG = "/World/SDG"; UsdGeom.Scope.Define(stage, SDG)
PARK = Gf.Vec3d(0.0, 0.0, -10.0)    # unused objects wait here, far below the floor and out of every view


def make(kind, path):
    if kind == "box":
        g = UsdGeom.Cube.Define(stage, path); g.CreateSizeAttr(1.0)
    elif kind == "sphere":
        g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(1.0)
    else:
        g = UsdGeom.Capsule.Define(stage, path); g.CreateAxisAttr("X"); g.CreateRadiusAttr(1.0); g.CreateHeightAttr(4.0)
    xf = UsdGeom.Xformable(g); t, r, s_ = xf.AddTranslateOp(), xf.AddRotateZOp(), xf.AddScaleOp()
    t.Set(PARK); r.Set(0.0); s_.Set(Gf.Vec3f(0.01, 0.01, 0.01))
    return dict(path=path, t=t, r=r, s=s_, col=g.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)]), kind=kind)


boxes = [make("box", f"{SDG}/box_{i}") for i in range(args.max_boxes)]
distractors = [make(k, f"{SDG}/distractor_{i}") for i, k in enumerate(DISTRACTOR_KINDS)]


def place(o, xyz, yaw_deg, scale, rgb_linear):
    o["t"].Set(Gf.Vec3d(*xyz)); o["r"].Set(float(yaw_deg)); o["s"].Set(Gf.Vec3f(*scale)); o["col"].Set([Gf.Vec3f(*rgb_linear)])


lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName()]
base_light = {str(p.GetPath()): (p.GetAttribute("inputs:intensity").Get() or 1.0) for p in lights if p.GetAttribute("inputs:intensity")}

# ---------- annotators: MUST be created before physics starts ----------
# Verified 2026-10-12 (Isaac 6.0, Replicator 1.13): a render product created after physics is initialised, or
# while the timeline plays, returns EMPTY data forever. It must be created AND render one frame before physics
# starts; after that it keeps delivering data with physics running.
rep.orchestrator.set_capture_on_play(False)
rp = rep.create.render_product(CAM, resolution=(W, H), name="sdg_wrist_cam")
annot = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ("rgb", "distance_to_image_plane")}
for a in annot.values():
    a.attach(rp)


def render_frame(max_tries=20):
    """Render until every annotator returns data. A new render product returns EMPTY data for its first
    frame(s) while the renderer builds its pipeline; reading too early gives a (0,)-shaped array."""
    for attempt in range(1, max_tries + 1):
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
        rgb_, dep_ = annot["rgb"].get_data(), annot["distance_to_image_plane"].get_data()
        if getattr(rgb_, "size", 0) and getattr(dep_, "size", 0):
            return rgb_[..., :3], dep_, None, attempt
        if attempt in (1, max_tries):
            print(f"[gen] frame attempt {attempt}: rgb {getattr(rgb_, 'shape', None)}, depth {getattr(dep_, 'shape', None)}", flush=True)
    raise SystemExit(f"annotators still empty after {max_tries} rendered frames")


_, _, _, warm = render_frame()
print(f"[gen] renderer warm: data ready after {warm} frame(s)", flush=True)


# ---------- physics: hold the robot at the capture pose ----------
# Physics runs through the normal timeline (Play) and full app updates. Verified 2026-10-12: stepping physics
# alone (SimulationManager.step) moves the joints but the renderer keeps a STALE pose for parts mounted on
# the links (the wrist camera), so images come from the wrong place. app.update() propagates the new poses.
import omni.timeline
omni.timeline.get_timeline_interface().play()
for _ in range(10):
    app.update()
arm = Articulation("/World/ur10")
dof = arm.get_dof_indices(cell.ARM_JOINTS)
Q0 = np.array(CFG["capture_poses"]["pick_overview"]["joints"], float)


TL = omni.timeline.get_timeline_interface()


def set_arm(q):
    global arm, dof
    arm = Articulation("/World/ur10"); dof = arm.get_dof_indices(cell.ARM_JOINTS)   # fresh handle: the timeline is restarted per layout
    arm.set_dof_positions(q[None, :], dof_indices=dof); arm.set_dof_position_targets(q[None, :], dof_indices=dof)
    for _ in range(30):
        app.update()


# ---------- randomisation ----------
def srgb_to_linear(c):
    """Palette values are screen colours (sRGB, what a photo of the object shows); USD displayColor is LINEAR light.
    Found 2026-10-19: using them unconverted made every box paler than intended ('black' 0.06 rendered mid-grey)."""
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def colour(palette):
    name, base, spread = palette[rng.integers(len(palette))]
    return name, srgb_to_linear(np.clip(np.array(base) + rng.normal(0, spread, 3), 0.02, 0.98))


def layout(dry=False):
    """A new random layout: parks every object, then places the ones this layout uses (timeline must be stopped).
    (Same random-number sequence as before 2026-10-19, so a seed gives the same layouts as box_v1.)
    dry=True draws exactly the same random numbers but touches nothing (for --resume)."""
    if not dry:
        for o in boxes + distractors:
            o["t"].Set(PARK)
    n = int(rng.integers(0, args.max_boxes + 1)); placed = []
    out = []
    for b in boxes[:n]:
        size = box_size * (1 + rng.uniform(-args.size_jitter, args.size_jitter, 3))
        for _ in range(50):                                  # no overlaps: centres at least one diagonal apart
            x, y = rng.uniform(*ZONE["x"]), rng.uniform(*ZONE["y"])
            if all(math.hypot(x - px, y - py) > (np.hypot(*size[:2]) + ps) / 2 + 0.003 for px, py, ps in placed):
                break
        else:
            continue
        yaw = rng.uniform(0, 360); placed.append((x, y, np.hypot(*size[:2])))
        fam, c = colour(PALETTES[args.palette])
        xyz = [x, y, ZONE["z"] + size[2] / 2]
        if not dry:
            place(b, xyz, yaw, size, c)
        out.append(dict(path=b["path"], world_xyz=xyz, yaw_deg=yaw, size=size.tolist(), colour=fam))
    for d in distractors:
        if rng.random() < 0.35:
            r_ = rng.uniform(0.008, 0.02); x, y = rng.uniform(*ZONE["x"]), rng.uniform(*ZONE["y"])
            if any(math.hypot(x - px, y - py) < ps / 2 + r_ * 3 for px, py, ps in placed):
                continue
            yaw_d = float(rng.uniform(0, 360)); col = srgb_to_linear(rng.uniform(0.05, 0.95, 3))
            if not dry:
                place(d, (x, y, ZONE["z"] + r_), yaw_d, (r_, r_, r_), col)
    for p in lights:
        a = p.GetAttribute("inputs:intensity")
        if a and str(p.GetPath()) in base_light:
            v = float(base_light[str(p.GetPath())] * rng.uniform(0.5, 1.6))
            if not dry:
                a.Set(v)
        c = p.GetAttribute("inputs:color")
        if c:
            v = Gf.Vec3f(*np.clip(1 + rng.normal(0, 0.06, 3), 0.8, 1.0))
            if not dry:
                c.Set(v)
    return out


# ---------- labels from geometry ----------
def usd_world(path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float).T


# The camera pose for labelling, from the robot (found 2026-10-12: Replicator's camera_params reports the STALE
# scene-file pose of a camera mounted on a moving link, here the all-joints-zero pose, not the physics pose):
#   world_T_camera = world_T_base x FK(joint angles from physics) x wrist_3_T_camera
# UR base = Isaac base_link turned 180 deg about Z, and Isaac wrist_3_link = UR tool0 (commissioning check, 0.00 mm).
# wrist_3_T_camera is rigid, so reading it once from the scene file is exact even though both poses there are stale.
WORLD_T_BASE = usd_world("/World/ur10/base_link") @ np.diag([-1.0, -1.0, 1.0, 1.0])
WRIST3_T_CAM = np.linalg.inv(usd_world("/World/ur10/wrist_3_link")) @ usd_world(CAM)
_cam = UsdGeom.Camera(stage.GetPrimAtPath(CAM))
FX = float(_cam.GetFocalLengthAttr().Get()) * W / float(_cam.GetHorizontalApertureAttr().Get())


def camera_to_world():
    q = np.asarray(arm.get_dof_positions().numpy()).reshape(-1)[:6]      # arm joints are dof 0-5 (verified)
    return WORLD_T_BASE @ kin.fk(q) @ WRIST3_T_CAM


def label_masks(depth, objs, W, H, xy_margin=0.0015, z_low=0.0015, z_high=0.003):
    """One boolean mask per placed object: pixels whose 3D point lies inside that object's known outline.
    Camera frame here is USD's: X right, Y up, looking along -Z."""
    C = camera_to_world(); fx, fy, cx, cy = FX, FX, W / 2.0, H / 2.0
    d = depth.astype(np.float64); ok = np.isfinite(d) & (d > 0)
    v, u = np.mgrid[0:H, 0:W]
    Pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d, np.ones_like(d)], -1)   # USD camera frame
    Pw = Pc @ C.T
    masks = []
    for o in objs:
        c, sz, yaw = np.array(o["world_xyz"]), np.array(o["size"]), math.radians(o["yaw_deg"])
        dx, dy, dz = Pw[..., 0] - c[0], Pw[..., 1] - c[1], Pw[..., 2] - c[2]
        lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        h = sz / 2
        masks.append(ok & (np.abs(lx) <= h[0] + xy_margin) & (np.abs(ly) <= h[1] + xy_margin)
                     & (dz >= -h[2] + z_low) & (dz <= h[2] + z_high))
    return masks, C


# ---------- output ----------
split_dir = os.path.join(args.out, args.split); os.makedirs(split_dir, exist_ok=True)
classes = list(CFG["objects"].keys())
coco = dict(info=dict(description="UR10 cell synthetic, Isaac Sim 6.0 Replicator", split=args.split, seed=args.seed,
                      palette=args.palette, created=time.strftime("%Y-%m-%d %H:%M:%S")),
            images=[], annotations=[],
            categories=[dict(id=0, name="objects", supercategory="none")] +
                       [dict(id=i + 1, name=c, supercategory="objects") for i, c in enumerate(classes)])
cat_id = {c: i + 1 for i, c in enumerate(classes)}


def save_coco():
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)


t_start = time.time(); ann_id = 0; start = 0
if args.resume:   # keep every frame already in the saved labels file; replay the random draws up to there
    old_path = os.path.join(split_dir, "_annotations.coco.json")
    if os.path.exists(old_path):
        old = json.load(open(old_path))
        start = len(old["images"])
        assert [im["file_name"] for im in old["images"]] == [f"{i:06d}.jpg" for i in range(start)], "saved frames are not 0..N-1"
        coco["images"], coco["annotations"] = old["images"], old["annotations"]
        ann_id = max((a_["id"] for a_ in old["annotations"]), default=-1) + 1
        coco["info"]["resumed_at_frame"] = start
        stale = sorted(n_ for n_ in os.listdir(split_dir) if n_[:6].isdigit() and int(n_[:6]) >= start)
        if stale:     # operator rule: back up before overwriting -> frames rendered but never recorded move aside first
            bdir = split_dir.rstrip("/") + f"_frames{start}+.{time.strftime('%Y%m%d-%H%M%S')}.bak"; os.makedirs(bdir, exist_ok=True)
            for n_ in stale:
                os.replace(os.path.join(split_dir, n_), os.path.join(bdir, n_))
            print(f"[gen] resume: {len(stale)} unrecorded file(s) from frame {start} on moved to {bdir}", flush=True)
    for f in range(start):
        if f % args.frames_per_episode == 0:
            q_episode = Q0 + np.radians(rng.uniform(-1.0, 1.0, 6))
        layout(dry=True)
    print(f"[gen] resume: {start} frames kept, continuing at frame {start} of {args.frames}", flush=True)
    t_start = time.time()
for f in range(start, args.frames):
    if args.max_new_frames and f - start >= args.max_new_frames:
        save_coco(); print(f"[gen] chunk done: {f - start} new frames this run, {len(coco['images'])} saved; exiting for a fresh process", flush=True)
        app.close(); sys.exit(3)
    if f % args.frames_per_episode == 0:
        q_episode = Q0 + np.radians(rng.uniform(-1.0, 1.0, 6))
    TL.stop(); app.update()               # objects are only changed while the timeline is STOPPED (see 'objects' above)
    objs = layout()
    app.update(); TL.play(); app.update()
    set_arm(q_episode)                    # stopping resets physics: pose the arm again
    for _ in range(args.settle_updates):  # the renderer shows a new layout only after several full updates
        app.update()                      # (found 2026-10-12: with 2 updates, images lagged the layout by frames)
    for _ in range(args.settle_frames):
        render_frame()
    rgb, depth, cam, _ = render_frame()
    masks, C = label_masks(depth, objs, W, H)
    if f == 0:   # transparency + self-check: the camera pose the renderer used must be where the robot put it
        print(f"[gen] first frame camera position (world) {np.round(C[:3, 3], 3).tolist()}, looking along "
              f"{np.round(-C[:3, 2], 3).tolist()}; height above the rollers {C[2, 3] - ZONE['z']:.3f} m "
              f"(capture pose expects ~0.6)", flush=True)
        print(f"[gen] first frame label pixels per placed box: {[int(m.sum()) for m in masks]}", flush=True)
    name = f"{f:06d}"
    cv2.imwrite(os.path.join(split_dir, name + ".jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(os.path.join(split_dir, name + "_depth.npy"), depth.astype(np.float16))
    img_id = len(coco["images"])
    coco["images"].append(dict(id=img_id, file_name=name + ".jpg", width=W, height=H))
    n_lab = 0
    for o, m in zip(objs, masks):
        cls = "box"
        mask = m.astype(np.uint8)
        area = int(mask.sum())
        if area < args.min_pixels:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]
        if not polys:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        coco["annotations"].append(dict(id=ann_id, image_id=img_id, category_id=cat_id[cls], bbox=[x, y, w, h],
                                        area=area, segmentation=polys, iscrowd=0)); ann_id += 1; n_lab += 1
    gt = []
    for o in objs:                                           # true poses in the robot base frame (scoring only)
        Tw = np.eye(4); Tw[:3, :3] = kin.rpy_to_R(0, 0, math.radians(o["yaw_deg"])); Tw[:3, 3] = o["world_xyz"]
        Tb = base_T_world @ Tw
        gt.append(dict(type="box", base_xyz=Tb[:3, 3].round(5).tolist(), base_yaw_deg=round(math.degrees(math.atan2(Tb[1, 0], Tb[0, 0])), 3),
                       size=o["size"], colour=o["colour"]))
    q_now = np.asarray(arm.get_dof_positions().numpy()).reshape(-1)[:6]   # what a real robot reports with each capture
    with open(os.path.join(split_dir, name + "_gt.json"), "w") as fh:
        json.dump(dict(joints=[round(float(v), 6) for v in q_now], objects=gt), fh)
    if f % 25 == 0 or f == args.frames - 1:
        save_coco()
        rate = (f + 1 - start) / (time.time() - t_start)
        print(f"[gen] {args.split} frame {f + 1}/{args.frames}: {len(objs)} boxes placed, {n_lab} labelled; {rate:.2f} frames/s", flush=True)

save_coco()
print(f"[gen] done: {len(coco['images'])} images, {len(coco['annotations'])} labelled instances -> {split_dir}", flush=True)
rep.orchestrator.wait_until_complete()
app.close()
