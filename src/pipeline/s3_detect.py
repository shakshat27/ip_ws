#!/usr/bin/env python3
"""Script 3 - DETECT. Finds objects in a capture: which pixels belong to which object.

Method 'geometry' (now): an object is anything standing on a taught surface. Depth pixels ->
3D points in the robot base frame (FK x hand-eye) -> keep points above a surface -> group
neighbouring pixels (connected components). It cannot name objects: class = 'unknown'.
Method 'rfdetr': the trained RF-DETR-Seg model (cell.yaml detector) on the RGB image: per object a class,
confidence, box and pixel mask. Same output format as 'geometry', with real class names. Each mask is
assigned to the taught surface under its depth points (script 4 needs the surface height), then CHECKED
against depth (cell.yaml detector.depth_check): kept only if most of its depth pixels stand above that
surface. Colour can be fooled (a picture, a shadow, a reflection, a render ghost); depth says whether
something physical is there. Rejected masks are listed in detections.json with the reason.
With detector.two_pass the model also runs on the colour-inverted image (dark boxes; see detect_rfdetr).
Needs the training environment (torch): run with `sipv`, not `sip`.
  IN : <run>/2_capture_<label>/ (rgb.png, depth.npy, camera_info.json, capture.json), cell.yaml
  OUT: <run>/3_detect_<label>/detections.json (per object: class, score, bbox, pixels, mask file,
       surface), mask_<i>.png, annotated.png

  python3 ~/ip_ws/src/pipeline/s3_detect.py [--run latest] [--capture 2_capture_pick_overview] [--method geometry|rfdetr]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, cv2
import cell, run_io, geometry as geo

NAME = "s3 detect"


def detect_geometry(depth, K, joints, cfg):
    dc = cfg["detection"]
    pts, valid = geo.backproject(depth, K, near=dc["near_cutoff"])
    B = geo.to_frame(pts, geo.base_T_camera(joints, cfg))
    found = []
    for sname, surf in cfg["surfaces"].items():
        above = B[..., 2] - surf["centre"][2]
        mask = (valid & geo.in_rectangle(B, surf) & (above > dc["min_height"]) & (above < dc["max_height"])).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area >= dc["min_pixels"]:
                found.append(dict(cls="unknown", score=1.0, bbox=[x, y, w, h], pixels=area, surface=sname,
                                  mask=(labels == i)))
    return found


_MODEL = {}


def load_model(cfg):
    """The trained detector, loaded once per process. Imported here so 'geometry' runs without torch."""
    if "m" not in _MODEL:
        try:
            import rfdetr
        except ImportError:
            raise SystemExit("method rfdetr needs the training environment: run with `sipv`")
        d = cfg["detector"]; cls = getattr(rfdetr, "RFDETRSeg" + d["size"].capitalize())
        _MODEL["m"] = cls(pretrain_weights=os.path.expanduser(d["model"]), trust_checkpoint=True)
        _MODEL["m"].predict(np.zeros((800, 1280, 3), np.uint8), threshold=0.99)   # warm-up: the first GPU call is ~40x slower
    return _MODEL["m"]


def surface_of(points, cfg):
    """Name of the taught surface under these 3D points (base frame): the one whose rectangle holds most of them,
    else the nearest by horizontal distance to their median."""
    best, best_n = None, 0
    for name, surf in cfg["surfaces"].items():
        n = int(geo.in_rectangle(points, surf).sum())
        if n > best_n:
            best, best_n = name, n
    if best is None and len(points):
        med = np.median(points, axis=0)
        best = min(cfg["surfaces"], key=lambda k: np.hypot(med[0] - cfg["surfaces"][k]["centre"][0], med[1] - cfg["surfaces"][k]["centre"][1]))
    return best


def depth_check(mask, B, valid, surf, cfg):
    """Does the depth camera see something standing on the surface where the model drew this mask?
    Same RealSense, same moment, depth aligned to colour: works identically on the real cell.
    Returns (keep, mask, raised): raised = share of the mask's valid depth pixels that stand between
    detection.min_height and max_height above the surface. With trim, mask pixels whose depth lies AT the
    surface are dropped (pixels with no depth reading are kept: no reading is not evidence)."""
    dc, ck = cfg["detection"], cfg["detector"]["depth_check"]
    above = B[..., 2] - surf["centre"][2]
    seen = mask & valid
    at_surface = seen & (above <= dc["min_height"])
    n_raised = int((seen & (above > dc["min_height"]) & (above < dc["max_height"])).sum())
    raised = n_raised / max(int(seen.sum()), 1)
    keep = n_raised >= ck["min_raised_pixels"] and raised >= ck["min_raised_fraction"]
    if keep and ck.get("trim"):
        mask = mask & ~at_surface
    return keep, mask, raised


def _candidates(det, source):
    names = det.data.get("class_name", ["object"] * len(det))
    return [(str(names[i]), float(det.confidence[i]), np.asarray(det.mask[i], bool), source) for i in range(len(det))]


def _overlap(a, b):
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(np.logical_or(a, b).sum() or 1)


def detect_rfdetr(rgb, depth, K, joints, cfg, model=None, use_depth_check=True, rejected=None, two_pass=None):
    """rgb: H x W x 3 RGB uint8. Returns the same list format as detect_geometry, plus 'raised' (depth support)
    and 'source' (which pass found it). Masks failing the depth check are left out; pass a list as `rejected`
    to receive them with the reason.
    Two-pass (cell.yaml detector.two_pass, suggested by Subhankar, 2026-10-21): the model also sees the
    colour-INVERTED image, where a black box looks white; an inverted-pass mask is added only where no
    normal-pass mask overlaps it. Both images go through the model in one batch (25 ms for both vs 46 ms as
    two calls). Measured on box_v2 test with the box_v1 model: black boxes 170 -> 190 of 192, no loss elsewhere."""
    model = model or load_model(cfg)
    dc = cfg["detector"]; two_pass = dc.get("two_pass", False) if two_pass is None else two_pass
    if two_pass:
        det_n, det_i = model.predict([rgb, 255 - rgb], threshold=dc["threshold"])
        cands = _candidates(det_n, "normal")
        cands += [c for c in _candidates(det_i, "inverted")
                  if all(_overlap(c[2], n[2]) < dc.get("two_pass_overlap", 0.3) for n in cands if n[3] == "normal")]
    else:
        cands = _candidates(model.predict(rgb, threshold=dc["threshold"]), "normal")
    if not cands:
        return []
    pts, valid = geo.backproject(depth, K, near=cfg["detection"]["near_cutoff"])
    B = geo.to_frame(pts, geo.base_T_camera(joints, cfg))
    found = []
    for cls, score, mask, source in cands:
        if not mask.any():
            continue
        sname, raised = surface_of(B[mask & valid], cfg), None
        if use_depth_check:
            if sname is None:
                if rejected is not None:
                    rejected.append(dict(cls=cls, score=score, pixels=int(mask.sum()), raised=None, source=source, reason="no depth on any taught surface"))
                continue
            keep, mask, raised = depth_check(mask, B, valid, cfg["surfaces"][sname], cfg)
            if not keep:
                if rejected is not None:
                    rejected.append(dict(cls=cls, score=score, pixels=int(mask.sum()), raised=raised, source=source, reason="nothing standing there in depth"))
                continue
        ys, xs = np.nonzero(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        found.append(dict(cls=cls, score=score, bbox=[x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
                          pixels=int(mask.sum()), surface=sname, raised=raised, source=source, mask=mask))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest"); ap.add_argument("--capture", help="capture step folder name")
    ap.add_argument("--method", default="geometry", choices=["geometry", "rfdetr"])
    a = ap.parse_args()
    cfg = cell.load(); cell.require(cfg, "hand_eye", "tool0_T_camera")
    for k, v in cfg["surfaces"].items():
        if v is None:
            raise SystemExit(f"cell.yaml: surface {k} is not taught yet")
    run = run_io.resolve(a.run)
    cap = os.path.join(run, a.capture) if a.capture else run_io.latest_step(run, "2_capture_")
    label = os.path.basename(cap).replace("2_capture_", "")
    depth = np.load(os.path.join(cap, "depth.npy")); K = run_io.read_json(os.path.join(cap, "camera_info.json"))["K"]
    capture = run_io.read_json(os.path.join(cap, "capture.json")); rgb = cv2.imread(os.path.join(cap, "rgb.png"))
    info = f"method {a.method}; surfaces {list(cfg['surfaces'])}; hand-eye from cell.yaml"
    if a.method == "rfdetr":
        info += f"; model {cfg['detector']['model']} (threshold {cfg['detector']['threshold']}, {'two-pass: normal + inverted' if cfg['detector'].get('two_pass') else 'single pass'})"
    run_io.show(NAME, "IN", [f"{cap}/", f"rgb + depth {depth.shape[1]}x{depth.shape[0]}, joints {np.round(capture['joints'], 4).tolist()}", info])
    rejected = []
    if a.method == "rfdetr":
        model = load_model(cfg)                                    # load outside the timing: done once per process
        t0 = time.perf_counter()
        found = detect_rfdetr(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth, K, capture["joints"], cfg, model, rejected=rejected)
    else:
        t0 = time.perf_counter()
        found = detect_geometry(depth, K, capture["joints"], cfg)
    ms = (time.perf_counter() - t0) * 1000

    out = run_io.step_dir(run, f"3_detect_{label}")
    dets = []
    for i, f in enumerate(found):
        mpath = f"mask_{i}.png"; cv2.imwrite(os.path.join(out, mpath), f["mask"].astype(np.uint8) * 255)
        x, y, w, h = f["bbox"]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (247, 162, 122), 2)
        cv2.putText(rgb, f"#{i} {f['cls']}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (247, 162, 122), 1, cv2.LINE_AA)
        dets.append(dict(id=i, cls=f["cls"], score=round(f["score"], 4), bbox=f["bbox"], pixels=f["pixels"], surface=f["surface"], source=f.get("source", "depth"),
                         raised=None if f.get("raised") is None else round(f["raised"], 3), mask=mpath))
    cv2.imwrite(os.path.join(out, "annotated.png"), rgb)
    run_io.write_json(os.path.join(out, "detections.json"), dict(capture=os.path.basename(cap), method=a.method, time_ms=ms, detections=dets,
                      rejected_by_depth=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rejected]))
    run_io.show(NAME, "OUT", [f"{out}/detections.json: {len(dets)} object(s) in {ms:.0f} ms"] +
                [f"#{d['id']} {d['cls']} ({d['score']:.2f}, {d['source']} pass) on {d['surface']}: bbox {d['bbox']}, {d['pixels']} px"
                 + (f", depth {d['raised'] * 100:.0f}% raised" if d["raised"] is not None else "") + f", {d['mask']}" for d in dets] +
                [f"rejected {r['cls']} ({r['score']:.2f}): {r['reason']}" + (f" ({r['raised'] * 100:.0f}% raised)" if r["raised"] is not None else "")
                 for r in rejected] + [f"{out}/annotated.png"])


if __name__ == "__main__":
    main()
