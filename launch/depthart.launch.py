import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("depthart_ros"),
        "config",
        "depthart.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Path to DepthART ROS parameter YAML file.",
        ),
        Node(
            package="depthart_ros",
            executable="depthart_node",
            name="depthart",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
