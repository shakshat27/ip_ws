"""Figures for the 2026-10-21 status report, built from real pipeline outputs (no simulator poses used to draw them)."""
import sys, os, json, numpy as np, cv2
sys.path[:0] = [os.path.expanduser("~/ip_ws/src/" + d) for d in ("tools", "lib", "pipeline")]
import cell, s3_detect
from eval_detectors import gt_masks, camera_K
OUT = os.path.expanduser("~/ip_ws/report/2026-10-21"); IP = os.path.expanduser("~/ip_ws")
GREEN, BLUE, RED, WHITE = (127, 209, 140), (247, 162, 122), (115, 107, 239), (255, 255, 255)

def title(im, text, h=46):
    bar = np.full((h, im.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (14, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.82, WHITE, 2, cv2.LINE_AA)
    return np.vstack([bar, im])

def pad(ims, gap=12):
    h = max(i.shape[0] for i in ims); out = []
    for k, i in enumerate(ims):
        if i.shape[0] < h: i = np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)])
        out.append(i)
        if k < len(ims) - 1: out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.hstack(out)

def outline(im, mask, colour, t=2):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cnts, -1, colour, t, cv2.LINE_AA)

# A: wrist camera at the pick pose + live detection (run 20261019_rfdetr_check)
run = f"{IP}/runs/20261019_rfdetr_check"
rgb = cv2.imread(f"{run}/2_capture_pick_overview/rgb.png"); m = cv2.imread(f"{run}/3_detect_pick_overview/mask_0.png", 0) > 0
obj = json.load(open(f"{run}/4_localize_pick_overview/objects.json"))["objects"][0]; det = json.load(open(f"{run}/3_detect_pick_overview/detections.json"))["detections"][0]
full = rgb.copy(); x, y, w, h = det["bbox"]; cv2.rectangle(full, (x - 30, y - 30), (x + w + 30, y + h + 30), BLUE, 3)
full = cv2.resize(full, (960, 600), interpolation=cv2.INTER_AREA)
y0, x0 = y + h // 2 - 60, x + w // 2 - 60; crop = rgb[y0:y0 + 120, x0:x0 + 120].copy(); mc = m[y0:y0 + 120, x0:x0 + 120]
crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_CUBIC); outline(crop, cv2.resize(mc.astype(np.uint8), (480, 480), interpolation=cv2.INTER_NEAREST), BLUE, 3)
info = np.full((120, 480, 3), 255, np.uint8)
for i, line in enumerate([f"box, confidence {det['score']:.2f}", f"x {obj['position'][0]:+.3f} m, y {obj['position'][1]:+.3f} m (robot base)", f"yaw {obj['yaw_deg']:.1f} deg, footprint {obj['footprint'][0]*1000:.0f} x {obj['footprint'][1]*1000:.0f} mm"]):
    cv2.putText(info, line, (12, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
figA = pad([title(full, "Wrist camera at the pick pose (0.60 m above the belt)"), title(np.vstack([crop, info]), "Detected box (mask outline)")])
cv2.imwrite(f"{OUT}/fig_camera_detection.png", figA)

# B: synthetic training images with their computed labels (box_v2 train)
D = f"{IP}/datasets/box_v2/train/"; c = json.load(open(D + "_annotations.coco.json")); by = {}
for a in c["annotations"]: by.setdefault(a["image_id"], []).append(a)
picks = [i for i in c["images"] if len(by.get(i["id"], [])) >= 3][3:400:66][:6]; tiles = []
for img in picks:
    im = cv2.imread(D + img["file_name"])
    for a in by[img["id"]]:
        for p in a["segmentation"]: cv2.polylines(im, [np.array(p, np.int32).reshape(-1, 2)], True, GREEN, 2, cv2.LINE_AA)
    tiles.append(cv2.resize(im[60:780, 100:1180], (540, 360), interpolation=cv2.INTER_AREA))
figB = title(np.vstack([pad(tiles[:3]), np.full((12, 540 * 3 + 24, 3), 255, np.uint8), pad(tiles[3:6])]), "Synthetic training images (Isaac Replicator); green = labels computed from depth and the known box outline")
cv2.imwrite(f"{OUT}/fig_dataset_samples.png", figB)

# C: touching boxes, classical (depth only) vs RF-DETR-Seg, box_v2 test frame
cfg = cell.load(); K, _ = camera_K(); model = s3_detect.load_model(cfg); T = f"{IP}/datasets/box_v2/test/"
ct = json.load(open(T + "_annotations.coco.json")); stem = "000116"; img = [i for i in ct["images"] if i["file_name"] == stem + ".jpg"][0]
bgr = cv2.imread(T + stem + ".jpg"); depth = np.load(T + stem + "_depth.npy").astype(np.float32); gt = json.load(open(T + stem + "_gt.json"))
geo_f = s3_detect.detect_geometry(depth, K, gt["joints"], cfg); rf_f = s3_detect.detect_rfdetr(bgr[..., ::-1].copy(), depth, K, gt["joints"], cfg, model)
ys, xs = np.nonzero(np.any([f["mask"] for f in rf_f], axis=0)); cy, cx = int(ys.mean()), int(xs.mean())
box = (max(cy - 170, 0), cy + 170, max(cx - 255, 0), cx + 255)
panels = []
for name, found in (("Classical (depth only): touching boxes merge", [f for f in geo_f if f["mask"][cy - 170:cy + 170, cx - 255:cx + 255].any()]), ("RF-DETR-Seg: each box separate", rf_f)):
    im = bgr.copy()
    for k, f in enumerate(found):
        col = [BLUE, (90, 200, 250), (200, 120, 230), (120, 200, 120), (60, 160, 255)][k % 5]; outline(im, f["mask"], col, 3)
    panels.append(title(cv2.resize(im[box[0]:box[1], box[2]:box[3]], (765, 510), interpolation=cv2.INTER_CUBIC), name))
cv2.imwrite(f"{OUT}/fig_touching_boxes.png", pad(panels))

# D: the render problem, same layout before (box_v1) and after (box_v2), test frame 24, labels in green
panels = []
for name, d in (("Before: pale colours, black boxes see-through", f"{IP}/datasets/box_v1/test/"), ("After: true colours, solid boxes", f"{IP}/datasets/box_v2/test/")):
    cc = json.load(open(d + "_annotations.coco.json")); im_ = [i for i in cc["images"] if i["file_name"] == "000024.jpg"][0]; im = cv2.imread(d + "000024.jpg")
    for a in cc["annotations"]:
        if a["image_id"] == im_["id"]:
            for p in a["segmentation"]: cv2.polylines(im, [np.array(p, np.int32).reshape(-1, 2)], True, GREEN, 2, cv2.LINE_AA)
    panels.append(title(cv2.resize(im[200:740, 150:1000], (765, 486), interpolation=cv2.INTER_AREA), name))
cv2.imwrite(f"{OUT}/fig_render_fix.png", pad(panels))
print("written:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
