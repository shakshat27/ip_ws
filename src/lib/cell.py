"""Cell configuration and shared constants. Every script reads cell.yaml through here."""
import os
import yaml

WS = os.path.expanduser("~/ip_ws")
CFG_PATH = os.path.join(WS, "config", "cell.yaml")
RUNS = os.path.join(WS, "runs")
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def load():
    return yaml.safe_load(open(CFG_PATH))


def require(cfg, *keys):
    """Stop with a clear message if a measured parameter is still missing (null) in cell.yaml."""
    node = cfg
    for k in keys:
        node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            raise SystemExit(f"cell.yaml: {'.'.join(keys)} is not set yet (must be measured, never taken from the simulator)")
    return node
