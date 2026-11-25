import os, yaml
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
