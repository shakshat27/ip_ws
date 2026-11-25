#!/usr/bin/env python3
"""Write a mini-PLC tag (tools/mini_plc.py) and optionally wait for the PLC to confirm the belt state.

  python3 ~/ip_ws/src/tools/plc_tag.py pick_request true --wait halted     # robot: hold the belt, wait until it is stopped
  python3 ~/ip_ws/src/tools/plc_tag.py pick_request false --wait running   # robot: done, belt may run (waits for the PLC)
  python3 ~/ip_ws/src/tools/plc_tag.py run_enable false                    # operator: stop the belt
Exit 0 = written (and confirmed when --wait); 1 = no PLC answered / not confirmed in plc.halt_timeout_s.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
import cell

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=["pick_request", "run_enable"])
    ap.add_argument("value", choices=["true", "false"])
    ap.add_argument("--wait", choices=["halted", "running"], default=None)
    a = ap.parse_args()
    c = cell.load().get("plc") or {}
    timeout = float(c.get("halt_timeout_s", 5.0)); value = a.value == "true"
    rclpy.init(); node = rclpy.create_node("plc_tag")
    state = {}
    node.create_subscription(String, "/plc/state", lambda m: state.update(json.loads(m.data)), LATCHED)
    pub = node.create_publisher(Bool, f"/plc/{a.tag}", LATCHED)
    t0 = time.time(); ok = False
    while time.time() - t0 < timeout:
        pub.publish(Bool(data=value))
        rclpy.spin_once(node, timeout_sec=0.05)
        if state.get(a.tag) == value and (a.wait is None or (a.wait == "halted" and state.get("halted"))
                                          or (a.wait == "running" and state.get("belt_running"))):
            ok = True; break
    msg = f"{a.tag} = {a.value}" + (f", belt {a.wait}" if a.wait else "")
    print(f"[plc tag] {msg}: {'confirmed' if ok else 'NOT confirmed by the PLC in %.1f s (is mini_plc.py running?)' % timeout} "
          f"({time.time() - t0:.2f} s; state {state or 'none'})", flush=True)
    node.destroy_node(); rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
