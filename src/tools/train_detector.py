#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg on our synthetic dataset (CV-4).

Run inside the training environment, with Isaac closed (it shares the GPU):
  source ~/ip_ws/.venv_cv/bin/activate
  python ~/ip_ws/src/tools/train_detector.py --data ~/ip_ws/datasets/box_v1 --size medium --epochs 30
  tensorboard --logdir ~/ip_ws/models          # in a second terminal, to watch the curves

  IN : <data>/train, <data>/valid (COCO: images + _annotations.coco.json, polygon masks)
  OUT: ~/ip_ws/models/<size>_<data>_<stamp>/  checkpoint_best_ema.pth (the model to use), checkpoints,
       TensorBoard logs, run.json (every setting, dataset counts, start/end time)
Every training setting is passed explicitly so a run can be repeated exactly.
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.expanduser("~/ip_ws/datasets/box_v1"))
ap.add_argument("--size", default="medium", choices=["nano", "small", "medium", "large"])
ap.add_argument("--epochs", type=int, default=30, help="passes over the training set")
ap.add_argument("--batch", type=int, default=4, help="images per step on the GPU")
ap.add_argument("--accum", type=int, default=4, help="steps whose gradients are summed before one update (effective batch = batch x accum)")
ap.add_argument("--lr", type=float, default=1e-4, help="learning rate: size of each weight update")
ap.add_argument("--resolution", type=int, default=None, help="input size, multiple of 56; default: the model's own (Seg-M 432)")
ap.add_argument("--early-stopping", action="store_true", help="stop when validation mAP stops improving")
ap.add_argument("--resume", default=None, help="checkpoint.pth of an interrupted run")
ap.add_argument("--out", default=None)
a = ap.parse_args()

data = os.path.expanduser(a.data)
counts = {}
for split in ("train", "valid", "test"):
    f = os.path.join(data, split, "_annotations.coco.json")
    if os.path.exists(f):
        c = json.load(open(f)); counts[split] = dict(images=len(c["images"]), objects=len(c["annotations"]),
                                                     classes=[k["name"] for k in c["categories"]])
for need in ("train", "valid"):
    if need not in counts:
        sys.exit(f"{data}/{need}/_annotations.coco.json missing: generate the dataset first")
if a.resolution and a.resolution % 56:
    sys.exit("--resolution must be a multiple of 56")
out = os.path.expanduser(a.out) if a.out else os.path.expanduser(
    f"~/ip_ws/models/{a.size}_{os.path.basename(data.rstrip('/'))}_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(out, exist_ok=True)
settings = dict(dataset_dir=data, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum, lr=a.lr,
                output_dir=out, tensorboard=True, early_stopping=a.early_stopping)
if a.resolution:
    settings["resolution"] = a.resolution
if a.resume:
    settings["resume"] = os.path.expanduser(a.resume)

print("[train] IN", flush=True)
for split, c in counts.items():
    print(f"    {split}: {c['images']} images, {c['objects']} labelled objects, classes {c['classes']}", flush=True)
print(f"    model RF-DETR-Seg {a.size}; settings {settings}", flush=True)

import torch
if not torch.cuda.is_available():
    sys.exit("PyTorch sees no GPU: activate ~/ip_ws/.venv_cv and close Isaac")
print(f"    GPU {torch.cuda.get_device_name(0)}, free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB", flush=True)
import rfdetr
cls = {"nano": rfdetr.RFDETRSegNano, "small": rfdetr.RFDETRSegSmall, "medium": rfdetr.RFDETRSegMedium, "large": rfdetr.RFDETRSegLarge}[a.size]

record = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), model=f"RFDETRSeg{a.size.capitalize()}", rfdetr_version=getattr(rfdetr, "__version__", "?"),
              torch=torch.__version__, gpu=torch.cuda.get_device_name(0), dataset_counts=counts, settings=settings)
json.dump(record, open(os.path.join(out, "run.json"), "w"), indent=1)
t0 = time.time()
model = cls()
model.train(**settings)
record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=round((time.time() - t0) / 60, 1))
json.dump(record, open(os.path.join(out, "run.json"), "w"), indent=1)
print("[train] OUT", flush=True)
print(f"    {out}/checkpoint_best_ema.pth  (best validation score; the file to use)", flush=True)
print(f"    {out}/run.json  ({record['minutes']} min)", flush=True)
