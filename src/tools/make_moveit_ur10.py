#!/usr/bin/env python3
"""Build MoveIt 2's configuration for OUR arm (UR10 CB3 + Robotiq 2F-85 + wrist D455) from the manufacturers' data.

  URDF   : UR's ur_description (xacro ur_type:=ur10; identical to our DH) + collision boxes on tool0 for the 2F-85 (open) and
           the D455 at the hand-eye mount: the same shapes the cuRobo model uses as spheres
  SRDF   : planning group `manipulator` (base_link -> tool0); collision pairs that touch by design (neighbours) or can never
           meet are disabled
  YAML   : KDL kinematics, OMPL (RRTConnect, then MoveIt's own path simplification), joint limits
  OUT    : ~/ip_ws/models/moveit_ur10/ (ur10_cell.urdf, ur10.srdf, kinematics.yaml, ompl.yaml, move_group.launch.py)
Run with the ROS environment:   source /opt/ros/jazzy/setup.bash; python3 ~/ip_ws/src/tools/make_moveit_ur10.py
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import cell, kinematics as kin

OUT = os.path.join(cell.WS, "models", "moveit_ur10")
LINKS = ["base_link", "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link", "tool_body", "tool_fingers_l", "tool_fingers_r", "camera_body"]


def box(name, size, xyz, rpy=(0, 0, 0)):
    return (f'<link name="{name}"><collision><origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'
            f'<geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry></collision></link>')


def fixed(name, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return (f'<joint name="{name}" type="fixed"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/></joint>')


def main():
    cfg = cell.load(); os.makedirs(OUT, exist_ok=True)
    from ament_index_python.packages import get_package_share_directory
    P = get_package_share_directory("ur_description")
    urdf = subprocess.run(["ros2", "run", "xacro", "xacro", os.path.join(P, "urdf", "ur.urdf.xacro"), "ur_type:=ur10", "name:=ur10"],
                          capture_output=True, text=True, check=True).stdout
    he = kin.T_from_cfg(cfg["hand_eye"]["tool0_T_camera"]); rpy = kin.R_to_rpy(he[:3, :3])
    ax = cfg["gripper"]["closing_axis_yaw_in_tool0"] or 0.0
    u = (np.cos(ax), np.sin(ax))
    add = [box("tool_body", (0.075, 0.087, 0.11), (0, 0, 0.055)),                    # 2F-85 coupling + body (datasheet size)
           box("tool_fingers_l", (0.03, 0.03, 0.07), (0.055 * u[0], 0.055 * u[1], 0.105)),   # open fingers + pads
           box("tool_fingers_r", (0.03, 0.03, 0.07), (-0.055 * u[0], -0.055 * u[1], 0.105)),
           box("camera_body", (0.124, 0.026, 0.029), (0, 0, 0)),                      # D455 (datasheet), at the hand-eye mount
           fixed("tool_body_joint", "tool0", "tool_body"), fixed("tool_fingers_l_joint", "tool0", "tool_fingers_l"),
           fixed("tool_fingers_r_joint", "tool0", "tool_fingers_r"),
           fixed("camera_joint", "tool0", "camera_body", he[:3, 3].tolist(), list(rpy))]
    urdf = urdf.replace("</robot>", "\n".join(add) + "\n</robot>")
    open(os.path.join(OUT, "ur10_cell.urdf"), "w").write(urdf)

    nb = [("base_link", "base_link_inertia"), ("base_link_inertia", "shoulder_link"), ("shoulder_link", "upper_arm_link"),
          ("upper_arm_link", "forearm_link"), ("forearm_link", "wrist_1_link"), ("wrist_1_link", "wrist_2_link"),
          ("wrist_2_link", "wrist_3_link"), ("wrist_3_link", "tool_body"), ("wrist_3_link", "tool_fingers_l"),
          ("wrist_3_link", "tool_fingers_r"), ("wrist_3_link", "camera_body"), ("tool_body", "tool_fingers_l"),
          ("tool_body", "tool_fingers_r"), ("tool_body", "camera_body"), ("tool_fingers_l", "tool_fingers_r"),
          ("tool_fingers_l", "camera_body"), ("tool_fingers_r", "camera_body"), ("wrist_2_link", "tool_body"),
          ("wrist_2_link", "camera_body"), ("wrist_1_link", "wrist_3_link"), ("shoulder_link", "wrist_1_link"),
          ("base_link", "shoulder_link"), ("base_link_inertia", "upper_arm_link"), ("shoulder_link", "forearm_link")]
    srdf = ['<?xml version="1.0"?>', '<robot name="ur10">',
            '<group name="manipulator"><chain base_link="base_link" tip_link="tool0"/></group>',
            '<group_state name="pick_view" group="manipulator">'
            + "".join(f'<joint name="{j}" value="{v}"/>' for j, v in zip(
                ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                cfg["capture_poses"]["pick_overview"]["joints"])) + '</group_state>']
    srdf += [f'<disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>' for a, b in nb] + ["</robot>"]
    open(os.path.join(OUT, "ur10.srdf"), "w").write("\n".join(srdf))

    open(os.path.join(OUT, "kinematics.yaml"), "w").write(
        "manipulator:\n  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin\n"
        "  kinematics_solver_search_resolution: 0.005\n  kinematics_solver_timeout: 0.05\n")
    open(os.path.join(OUT, "ompl.yaml"), "w").write(
        "planning_plugins:\n  - ompl_interface/OMPLPlanner\n"
        "request_adapters:\n  - default_planning_request_adapters/ResolveConstraintFrames\n"
        "  - default_planning_request_adapters/ValidateWorkspaceBounds\n"
        "  - default_planning_request_adapters/CheckStartStateBounds\n"
        "  - default_planning_request_adapters/CheckStartStateCollision\n"
        "response_adapters:\n  - default_planning_response_adapters/ValidateSolution\n"
        "planner_configs:\n  RRTConnectkConfigDefault:\n    type: geometric::RRTConnect\n    range: 0.0\n"
        "manipulator:\n  default_planner_config: RRTConnectkConfigDefault\n  planner_configs:\n    - RRTConnectkConfigDefault\n"
        "  projection_evaluator: joints(shoulder_pan_joint,shoulder_lift_joint)\n  longest_valid_segment_fraction: 0.005\n")
    open(os.path.join(OUT, "move_group.launch.py"), "w").write('''import os, yaml
from launch import LaunchDescription
from launch_ros.actions import Node
D = os.path.dirname(os.path.abspath(__file__))
rd = lambda n: open(os.path.join(D, n)).read()
ry = lambda n: yaml.safe_load(open(os.path.join(D, n)))
def generate_launch_description():
    return LaunchDescription([Node(package="moveit_ros_move_group", executable="move_group", output="screen", parameters=[
        {"robot_description": rd("ur10_cell.urdf"), "robot_description_semantic": rd("ur10.srdf"),
         "robot_description_kinematics": ry("kinematics.yaml"), "planning_pipelines": {"pipeline_names": ["ompl"], "default_planning_pipeline": "ompl"},
         "ompl": ry("ompl.yaml"), "publish_robot_description_semantic": False, "publish_planning_scene": False,
         "publish_geometry_updates": False, "publish_state_updates": False, "publish_transforms_updates": False,
         "monitor_dynamics": False, "allow_trajectory_execution": False, "capabilities": ""}])])
''')
    print(f"-> {OUT}/ (ur10_cell.urdf with {len(LINKS)} links, ur10.srdf, kinematics.yaml, ompl.yaml, move_group.launch.py)")


if __name__ == "__main__":
    main()
