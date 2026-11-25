#!/usr/bin/env python3
"""Teach a capture pose: store the arm's current joints (/joint_states) under cell.yaml capture_poses.<name>.

The way a real cell is taught: the operator jogs the arm (s0 page, jog.py) until the camera sees what it should, then
saves the joints. An existing entry gets its joints replaced (its other fields stay); a new name is added at the end of
capture_poses. Backup <file>.<stamp>.bak next to cell.yaml.
  IN : /joint_states (arm still)
  OUT: cell.yaml capture_poses.<name>.joints

  source ~/ip_ws/env.sh
  python3 ~/ip_ws/src/tools/teach_pose.py --name drop_overview [--looks-at basket] [--purpose "find the basket opening"]
"""
import argparse, os, re, shutil, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--looks-at", default=None); ap.add_argument("--purpose", default=None)
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", a.name):
        raise SystemExit("name: lower case letters, digits and _ only")
    import rclpy
    from ros_io import RobotIO
    rclpy.init(); io = RobotIO("teach_pose")
    try:
        if not io.wait_for(lambda: io.q is not None, 8):
            raise SystemExit("no /joint_states: is Isaac playing on domain 33?")
        if not io.wait_still(0.005, hold=0.5, timeout=6):
            raise SystemExit("the arm is moving: wait until it is still")
        q = [round(float(v), 4) for v in io.q]
    finally:
        rclpy.shutdown()
    path = os.path.join(cell.WS, "config", "cell.yaml")
    txt = open(path).read()
    new, action = update(txt, a.name, q, a.looks_at, a.purpose, time.strftime("%Y-%m-%d %H:%M"))
    bak = path + time.strftime(".%Y%m%d-%H%M%S.bak"); shutil.copy2(path, bak)
    open(path, "w").write(new)
    print(f"teach: {action} capture_poses.{a.name}: joints {q} rad (deg {np.round(np.degrees(q), 1).tolist()}); backup {os.path.basename(bak)}", flush=True)


def update(txt, name, q, looks_at=None, purpose=None, stamp=""):
    """cell.yaml text with capture_poses.<name>.joints set (entry replaced in place, or appended to the section)."""
    sec = re.search(r"^capture_poses:.*?\n(?=\S)", txt, re.S | re.M)
    if not sec:
        raise SystemExit("cell.yaml has no capture_poses section; nothing written")
    block = sec.group(0)
    jl = f"    joints: {q}   # TAUGHT {stamp} by operator (jog + teach)"
    within = rf"(^  {name}:\n(?:(?!  \S).*\n)*?)"                 # lines of this entry, up to the next entry
    if re.search(rf"^  {name}:\s*$", block, re.M):
        new = re.sub(within + r"    joints:.*\n", lambda m: m.group(1) + jl + "\n", block, count=1, flags=re.M)
        if new == block:                                          # entry without a joints line: add one under the name
            new = re.sub(rf"^  {name}:\n", f"  {name}:\n{jl}\n", block, count=1, flags=re.M)
        for key, val in (("looks_at", looks_at), ("purpose", purpose)):
            if val is not None:
                new = re.sub(within + rf"    {key}:.*\n", lambda m: m.group(1) + f"    {key}: {val}\n", new, count=1, flags=re.M)
        action = "replaced the joints of"
    else:
        extra = "".join(f"    {k}: {v}\n" for k, v in (("looks_at", looks_at), ("purpose", purpose)) if v is not None)
        new = block.rstrip("\n") + f"\n  {name}:\n{jl}\n{extra}\n"
        action = "added"
    return txt.replace(block, new, 1), action


if __name__ == "__main__":
    main()
