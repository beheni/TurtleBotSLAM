from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    viz_py = Node(
        package='turtleSLAM',
        executable='parse_lidar',
        name='parse_lidar',
        output='screen',
    )

    return LaunchDescription([
        Node(
            package='slam_cpp',
            executable='coordinate_converter',
            name='coordinate_converter',
            output='screen'),
        Node(
            package='slam_cpp',
            executable='landmark_corrector',
            name='landmark_corrector',
            output='screen'),
        viz_py
    ])
