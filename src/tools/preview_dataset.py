#!/usr/bin/env python3
"""Draw a dataset's labels over its images, to check them by eye.

For each image: every labelled instance's polygon mask (filled, translucent), its bounding box and
class name. Writes one sheet of thumbnails per call.
  IN : <dataset>/<split>/_annotations.coco.json + images
  OUT: <dataset>/<split>/_preview.png

  python3 ~/ip_ws/src/tools/preview_dataset.py ~/ip_ws/datasets/box_v1/smoke [--n 12]
"""
import argparse, json, os
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("split_dir"); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
d = os.path.expanduser(a.split_dir); coco = json.load(open(os.path.join(d, "_annotations.coco.json")))
cats = {c["id"]: c["name"] for c in coco["categories"]}
by_img = {}
for an in coco["annotations"]:
    by_img.setdefault(an["image_id"], []).append(an)
print(f"[preview] IN  {d}: {len(coco['images'])} images, {len(coco['annotations'])} instances, classes {sorted(set(cats.values()))}")
tiles = []
for im in coco["images"][:a.n]:
    img = cv2.imread(os.path.join(d, im["file_name"])); over = img.copy()
    for an in by_img.get(im["id"], []):
        for poly in an["segmentation"]:
            cv2.fillPoly(over, [np.array(poly, np.int32).reshape(-1, 2)], (247, 162, 122))
        x, y, w, h = an["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (247, 162, 122), 2)
        cv2.putText(img, cats[an["category_id"]], (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (247, 162, 122), 2, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    img = cv2.resize(img, (640, 400))
    cv2.putText(img, f"{im['file_name']}  {len(by_img.get(im['id'], []))} labelled", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(img)
if len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
out = os.path.join(d, "_preview.png")
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
print(f"[preview] OUT {out}")
