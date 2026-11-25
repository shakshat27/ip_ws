#!/usr/bin/env python3
"""Mini PLC for the UR10 cell: owns the belt. The robot side never drives the belt; it writes a TAG and the PLC decides.

Tags (ROS 2, domain 33; tag writes are latched, so a write sent before the PLC starts is still seen):
  IN   /plc/pick_request  std_msgs/Bool     robot: true = I am picking, hold the belt; false = done, run
  IN   /plc/run_enable    std_msgs/Bool     operator: false stops the belt whatever the robot says (default true)
  OUT  /plc/belt_cmd      std_msgs/Float64  belt surface speed (m/s, +x world); Isaac graph /Graphs/BeltPLC writes it
                                            into /World/Conveyors/Plane physics:velocity
  OUT  /plc/state         std_msgs/String   JSON: tags, belt_running, halted (belt commanded 0 for at least one scan)
Scan (cell.yaml plc.scan_s): belt_cmd = plc.belt_speed if run_enable and not pick_request else 0. Outputs are re-sent
every scan, so Isaac picks the speed up again after a restart.

  source ~/ip_ws/env.sh; python3 ~/ip_ws/src/tools/mini_plc.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String
import cell

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class MiniPLC(Node):
    def __init__(self, cfg):
        super().__init__("mini_plc")
        c = cfg.get("plc") or {}
        self.speed, self.scan = float(c.get("belt_speed", 0.03)), float(c.get("scan_s", 0.05))
        self.tags = dict(pick_request=False, run_enable=True)
        self.out = dict(belt_cmd=None, belt_running=False, halted=False, changed=time.time())
        for tag in self.tags:
            self.create_subscription(Bool, f"/plc/{tag}", lambda m, t=tag: self.write(t, m.data), LATCHED)
        self.cmd_pub = self.create_publisher(Float64, "/plc/belt_cmd", 10)
        self.state_pub = self.create_publisher(String, "/plc/state", LATCHED)
        self.create_timer(self.scan, self.cycle)
        self.get_logger().info(f"mini PLC up: belt {self.speed} m/s, scan {self.scan * 1000:.0f} ms")

    def write(self, tag, value):
        if self.tags[tag] != bool(value):
            self.get_logger().info(f"tag {tag} -> {bool(value)}")
        self.tags[tag] = bool(value)

    def cycle(self):
        cmd = self.speed if self.tags["run_enable"] and not self.tags["pick_request"] else 0.0
        if cmd != self.out["belt_cmd"]:
            self.get_logger().info(f"belt {'RUN' if cmd else 'HALT'} ({cmd} m/s)")
            self.out.update(belt_cmd=cmd, changed=time.time(), halted=False)
        elif cmd == 0.0:
            self.out["halted"] = True                     # the zero has been sent for at least one full scan
        self.out["belt_running"] = cmd != 0.0
        self.cmd_pub.publish(Float64(data=cmd))
        self.state_pub.publish(String(data=json.dumps(dict(self.tags, belt_cmd=cmd, belt_running=self.out["belt_running"],
                                                           halted=self.out["halted"], since_s=round(time.time() - self.out["changed"], 2)))))


def main():
    rclpy.init()
    node = MiniPLC(cell.load())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
