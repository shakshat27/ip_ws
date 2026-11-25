#!/usr/bin/env python3
"""REFEREE TOOL (scoring only). Compares detectors on a held-out dataset split.

For every image: run each detection method exactly as script 3 does, then
  2D  - match predicted masks to the true masks (IoU >= --iou): found (TP), false (FP), missed (FN), mask IoU
  3D  - run script 4's pose measurement on each detection (camera pose from the image's joint angles x our
        CALIBRATED hand-eye, exactly as on the robot) and compare with the true box pose: position error in
        mm (horizontal, height) and yaw error in degrees (wrapped by the box's symmetry); time per image.
Ground truth (masks from the COCO file, poses from _gt.json) is used only here, to score.

  IN : <data>/ (images, _annotations.coco.json, NNNNNN_depth.npy, NNNNNN_gt.json), cell.yaml, the trained model
  OUT: ~/ip_ws/evals/<stamp>_<split>/  summary.json, detections.csv, worst/ (images of misses and false alarms)

  sipv
  python ~/ip_ws/src/tools/eval_detectors.py --data ~/ip_ws/datasets/box_v1/test [--methods geometry rfdetr] [--limit 50]
"""
import argparse, csv, glob, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), os.path.join(HERE, "..", "pipeline")]
import numpy as np, cv2
import cell, geometry as geo, run_io
import s3_detect, s4_localize

NAME = "eval"


def camera_K():
    """Lens model as the real camera reports it: from the newest pipeline capture's camera_info.json."""
    files = sorted(glob.glob(os.path.join(cell.RUNS, "*", "2_capture_*", "camera_info.json")), key=os.path.getmtime)
    if files:
        return run_io.read_json(files[-1])["K"], files[-1]
    return [[634.1, 0, 640.0], [0, 634.1, 400.0], [0, 0, 1]], "default D455 1280x800 (no capture found)"


def gt_masks(coco, img, H, W):
    out = []
    for a in coco["by_img"].get(img["id"], []):
        m = np.zeros((H, W), np.uint8)
        for poly in a["segmentation"]:
            cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
        out.append(m.astype(bool))
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(np.logical_or(a, b).sum() or 1)


PERTURB = {   # colour-image degradations a real camera produces; depth is never touched (2026-10-21)
    "dark": "brightness x0.45 (underexposed)", "bright": "brightness x1.6, clipped (overexposed)",
    "noise": "Gaussian sensor noise, sigma 12 grey levels", "blur": "Gaussian blur, sigma 2 px (defocus / motion)",
    "lowcontrast": "contrast halved around mid-grey"}


def perturb(rgb, kind, seed):
    x = rgb.astype(np.float32)
    if kind == "dark":
        x = x * 0.45
    elif kind == "bright":
        x = x * 1.6
    elif kind == "noise":
        x = x + np.random.default_rng(seed).normal(0, 12, x.shape)
    elif kind == "blur":
        x = cv2.GaussianBlur(x, (0, 0), 2)
    elif kind == "lowcontrast":
        x = 128 + (x - 128) * 0.5
    return np.clip(x, 0, 255).astype(np.uint8)


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return dict(n=int(len(v)), mean=round(float(v.mean()), 3), sd=round(float(v.std()), 3),
                median=round(float(np.median(v)), 3), p95=round(float(np.percentile(v, 95)), 3), max=round(float(v.max()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/ip_ws/datasets/box_v1/test"))
    ap.add_argument("--methods", nargs="+", default=["geometry", "rfdetr"], choices=["geometry", "rfdetr", "rfdetr-1pass", "rfdetr-raw"],
                    help="rfdetr = model + depth check, passes as cell.yaml says (detector.two_pass); rfdetr-1pass = depth check, "
                         "normal image only; rfdetr-raw = model alone, normal image only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iou", type=float, default=0.5, help="mask overlap needed to count as found")
    ap.add_argument("--match-dist", type=float, default=0.03, help="m; a pose is compared with the true box nearer than this")
    ap.add_argument("--perturb", default=None, choices=sorted(PERTURB), help="degrade the colour image first (stress test; depth untouched)")
    ap.add_argument("--model", default=None, help="checkpoint to evaluate instead of cell.yaml detector.model (cell.yaml is not changed)")
    a = ap.parse_args()
    cfg = cell.load(); cell.require(cfg, "hand_eye", "tool0_T_camera")
    if a.model:
        cfg["detector"]["model"] = a.model
    data = os.path.expanduser(a.data)
    coco = json.load(open(os.path.join(data, "_annotations.coco.json")))
    coco["by_img"] = {}
    for an in coco["annotations"]:
        coco["by_img"].setdefault(an["image_id"], []).append(an)
    images = coco["images"][: a.limit] if a.limit else coco["images"]
    K, k_src = camera_K()
    model = s3_detect.load_model(cfg) if any(m.startswith("rfdetr") for m in a.methods) else None
    run_io.show(NAME, "IN", [f"{data}: {len(images)} images, {sum(len(coco['by_img'].get(i['id'], [])) for i in images)} labelled boxes",
                             f"methods {a.methods}; match: mask IoU >= {a.iou}, pose within {a.match_dist * 1000:.0f} mm",
                             f"camera lens from {k_src}; hand-eye and surfaces from cell.yaml"] +
                ([f"model {cfg['detector']['model']} (threshold {cfg['detector']['threshold']})"] if model else []))
    out = os.path.join(os.path.expanduser("~/ip_ws/evals"), time.strftime("%Y%m%d_%H%M%S") + "_" + os.path.basename(data.rstrip("/")) + (f"_{a.perturb}" if a.perturb else ""))
    os.makedirs(os.path.join(out, "worst"), exist_ok=True)
    rows, agg = [], {m: dict(tp=0, fp=0, fn=0, ious=[], exy=[], ez=[], eyaw=[], ms=[], typed_box=0, posed=0, col_all={}, col_hit={}) for m in a.methods}
    worst_saved = {m: 0 for m in a.methods}
    for n, img in enumerate(images):
        stem = os.path.splitext(img["file_name"])[0]
        bgr = cv2.imread(os.path.join(data, img["file_name"]))
        if a.perturb:
            bgr = perturb(bgr, a.perturb, n)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(data, stem + "_depth.npy")).astype(np.float32)
        gt = json.load(open(os.path.join(data, stem + "_gt.json")))
        joints = gt["joints"]; H, W = depth.shape
        gms = gt_masks(coco, img, H, W)
        pts, valid = geo.backproject(depth, K, near=cfg["detection"]["near_cutoff"])
        B = geo.to_frame(pts, geo.base_T_camera(joints, cfg))
        for m in a.methods:
            t0 = time.perf_counter()
            if m.startswith("rfdetr"):
                found = s3_detect.detect_rfdetr(rgb, depth, K, joints, cfg, model, use_depth_check=(m != "rfdetr-raw"),
                                                two_pass=None if m == "rfdetr" else False)
            else:
                found = s3_detect.detect_geometry(depth, K, joints, cfg)
            ms = (time.perf_counter() - t0) * 1000; agg[m]["ms"].append(ms)
            # 2D matching, greedy by IoU
            pairs = sorted(((iou(f["mask"], g), i, j) for i, f in enumerate(found) for j, g in enumerate(gms)), reverse=True)
            used_f, used_g, match = set(), set(), {}
            for v, i, j in pairs:
                if v < a.iou:
                    break
                if i not in used_f and j not in used_g:
                    used_f.add(i); used_g.add(j); match[i] = v
            agg[m]["tp"] += len(match); agg[m]["fp"] += len(found) - len(match); agg[m]["fn"] += len(gms) - len(match)
            if len(gms) == len(gt["objects"]):     # one label per placed box (checked): label j = box j, so colour is exact
                for j in range(len(gms)):
                    col = gt["objects"][j].get("colour", "?"); agg[m]["col_all"][col] = agg[m]["col_all"].get(col, 0) + 1
                    if j in used_g:
                        agg[m]["col_hit"][col] = agg[m]["col_hit"].get(col, 0) + 1
            agg[m]["ious"] += list(match.values())
            # 3D pose of every detection, compared with the nearest true box
            for i, f in enumerate(found):
                P = B[f["mask"] & valid]
                row = dict(method=m, image=img["file_name"], det=i, score=round(f["score"], 3), pixels=f["pixels"],
                           matched_2d=i in match, mask_iou=round(match.get(i, 0.0), 3), ms=round(ms, 1))
                if f["surface"] and len(P) >= 20:
                    meas = s4_localize.measure(P, cfg["surfaces"][f["surface"]]["centre"][2])
                    typ = f["cls"] if f["cls"] != "unknown" else (s4_localize.match_type(meas, cfg["objects"]) or ("unknown",))[0]
                    agg[m]["posed"] += 1; agg[m]["typed_box"] += typ == "box"
                    near = min(gt["objects"], key=lambda o: math.hypot(o["base_xyz"][0] - meas["position"][0], o["base_xyz"][1] - meas["position"][1]), default=None)
                    if near is not None:
                        d = np.array(meas["position"]) - np.array(near["base_xyz"])
                        if math.hypot(d[0], d[1]) <= a.match_dist:
                            exy, ez = math.hypot(d[0], d[1]) * 1000, d[2] * 1000
                            eyaw = abs(geo.wrap(meas["yaw_deg"] - near["base_yaw_deg"], cfg["objects"]["box"].get("yaw_symmetry_deg", 90)))
                            agg[m]["exy"].append(exy); agg[m]["ez"].append(ez); agg[m]["eyaw"].append(eyaw)
                            row.update(err_xy_mm=round(exy, 2), err_z_mm=round(ez, 2), err_yaw_deg=round(eyaw, 2), typed=typ)
                rows.append(row)
            # keep a few images of mistakes to look at
            if (len(found) != len(match) or len(gms) != len(match)) and worst_saved[m] < 12:
                vis = bgr.copy()
                for g in gms:
                    cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, (127, 209, 140), 2)                     # true boxes: green
                for i, f in enumerate(found):
                    cnts, _ = cv2.findContours(f["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, (247, 162, 122) if i in match else (115, 107, 239), 2)   # found: blue, false: red
                cv2.imwrite(os.path.join(out, "worst", f"{m}_{stem}.jpg"), vis); worst_saved[m] += 1
        if (n + 1) % 50 == 0:
            print(f"[{NAME}] {n + 1}/{len(images)} images", flush=True)

    summary = dict(data=data, images=len(images), methods={}, settings=vars(a), camera_lens=k_src,
                   model=cfg["detector"]["model"] if model else None, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    for m, g in agg.items():
        p = g["tp"] / max(1, g["tp"] + g["fp"]); r = g["tp"] / max(1, g["tp"] + g["fn"])
        summary["methods"][m] = dict(found=g["tp"], false=g["fp"], missed=g["fn"], precision=round(p, 4), recall=round(r, 4),
                                     mask_iou=stats(g["ious"]), err_xy_mm=stats(g["exy"]), err_z_mm=stats(g["ez"]),
                                     err_yaw_deg=stats(g["eyaw"]), ms=stats(g["ms"]),
                                     typed_as_box=f"{g['typed_box']}/{g['posed']}",
                                     recall_by_colour={c: f"{g['col_hit'].get(c, 0)}/{n_}" for c, n_ in sorted(g["col_all"].items())})
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(os.path.join(out, "detections.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    lines = [f"{out}/summary.json, detections.csv, worst/ (green = true, blue = found, red = false)"]
    for m, s in summary["methods"].items():
        f = lambda st, k="mean": "-" if st is None else st[k]
        lines += [f"{m:8s} found {s['found']}, missed {s['missed']}, false {s['false']}  (precision {s['precision']}, recall {s['recall']}), "
                  f"mask IoU {f(s['mask_iou'])}, typed as box {s['typed_as_box']}",
                  f"{'':8s} position error mm: horizontal mean {f(s['err_xy_mm'])} / p95 {f(s['err_xy_mm'], 'p95')}, "
                  f"height mean {f(s['err_z_mm'])}; yaw deg mean {f(s['err_yaw_deg'])} / p95 {f(s['err_yaw_deg'], 'p95')}; "
                  f"time ms median {f(s['ms'], 'median')}"]
    run_io.show(NAME, "OUT", lines)


if __name__ == "__main__":
    main()
