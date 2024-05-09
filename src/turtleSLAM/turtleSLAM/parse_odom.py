import rclpy
import numpy as np
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ParseOdom(Node):
    def __init__(self):
        super().__init__('parse_odom')
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, Odometry, '/odom', qos_profile=qos.qos_profile_sensor_data), 
                                             Subscriber(self, LaserScan, '/scan')], 10, 0.25)
        self.subscription.registerCallback(self.callback)
    
    def callback(self, odometry, lidar):
        v = odometry.twist.twist.linear.x
        w = odometry.twist.twist.angular.z
        interval = 80
        angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)
        ranges = np.array(lidar.ranges)
        ranges = np.array(list(zip(angles, ranges)))
        ranges = ranges[np.where(ranges[:,1] != np.inf)]
        self.get_logger().info(f'Odometry: v={v:.3f}, w={w:.3f}')
        self.get_logger().info(f'Lidar: {ranges[::interval]}')


def main():
    rclpy.init()
    parse_odom = ParseOdom()
    rclpy.spin(parse_odom)
    parse_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
