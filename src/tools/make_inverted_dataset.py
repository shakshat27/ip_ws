#!/usr/bin/env python3
"""Training-time version of the two-pass idea (Subhankar, 2026-10-21): add a colour-INVERTED copy of every image.

Inverting turns white and light-grey boxes into dark ones, so the model sees dark boxes without regenerating any
scene. Labels are unchanged (inversion moves no pixel). Train and valid get originals + inverted copies (valid too,
so the best checkpoint is chosen on both); test stays the untouched source test split, so black remains a colour
the model never trained on and the test still measures generalisation.
  IN : <src>/{train,valid,test}/ (COCO: _annotations.coco.json + NNNNNN.jpg)
  OUT: <out>/{train,valid}/  originals as symlinks + inv_NNNNNN.jpg + merged _annotations.coco.json;
       <out>/test -> <src>/test (symlink)

  python3 ~/ip_ws/src/tools/make_inverted_dataset.py --src ~/ip_ws/datasets/box_v2 --out ~/ip_ws/datasets/box_v2_inv
"""
import argparse, copy, json, os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.expanduser("~/ip_ws/datasets/box_v2"))
ap.add_argument("--out", default=os.path.expanduser("~/ip_ws/datasets/box_v2_inv"))
a = ap.parse_args()
src, out = os.path.abspath(os.path.expanduser(a.src)), os.path.abspath(os.path.expanduser(a.out))
if os.path.exists(out):
    raise SystemExit(f"{out} exists: choose another --out (nothing is overwritten)")
os.makedirs(out)
for split in ("train", "valid"):
    sd, od = os.path.join(src, split), os.path.join(out, split); os.makedirs(od)
    coco = json.load(open(os.path.join(sd, "_annotations.coco.json")))
    merged = copy.deepcopy(coco); merged["info"] = dict(coco.get("info", {}), inverted_copies=True, source=sd)
    next_img = max(i["id"] for i in coco["images"]) + 1; next_ann = max((x["id"] for x in coco["annotations"]), default=-1) + 1
    by_img = {}
    for x in coco["annotations"]:
        by_img.setdefault(x["image_id"], []).append(x)
    for img in coco["images"]:
        os.symlink(os.path.join(sd, img["file_name"]), os.path.join(od, img["file_name"]))
        inv_name = "inv_" + img["file_name"]
        cv2.imwrite(os.path.join(od, inv_name), 255 - cv2.imread(os.path.join(sd, img["file_name"])), [cv2.IMWRITE_JPEG_QUALITY, 95])
        merged["images"].append(dict(img, id=next_img, file_name=inv_name))
        for x in by_img.get(img["id"], []):
            merged["annotations"].append(dict(x, id=next_ann, image_id=next_img)); next_ann += 1
        next_img += 1
    json.dump(merged, open(os.path.join(od, "_annotations.coco.json"), "w"))
    print(f"[invert] {split}: {len(coco['images'])} -> {len(merged['images'])} images, {len(coco['annotations'])} -> {len(merged['annotations'])} labels", flush=True)
os.symlink(os.path.join(src, "test"), os.path.join(out, "test"))
print(f"[invert] test -> {os.path.join(src, 'test')} (untouched, held-out colours)\n[invert] OUT {out}")
