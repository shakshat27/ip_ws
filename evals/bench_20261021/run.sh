#!/usr/bin/env bash
# Two-pass vs single-pass benchmark, 2026-10-21: new model (box_v2 + inverted copies) on clean and degraded images.
source ~/ip_ws/env.sh >/dev/null 2>&1; source ~/ip_ws/.venv_cv/bin/activate
E=~/ip_ws/src/tools/eval_detectors.py; D=~/ip_ws/datasets/box_v2
for args in "--data $D/test" "--data $D/valid" "--data $D/test --perturb dark" "--data $D/test --perturb bright" "--data $D/test --perturb noise" "--data $D/test --perturb blur" "--data $D/test --perturb lowcontrast"; do
  echo "=== $args"; python $E $args --methods rfdetr-1pass rfdetr 2>&1 | grep -E "^\s+/|found" 
done
