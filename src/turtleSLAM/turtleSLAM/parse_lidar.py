import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np
from rclpy import qos

class ParseLidar(Node):
    def __init__(self):
        super().__init__('parse_odom')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos_profile=qos.qos_profile_sensor_data)
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plt.ion()  # Enable interactive mode for continuous updating
        plt.show()

    
    def lidar_callback(self, scan_msg):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        ranges = np.array(scan_msg.ranges)
        valid_ranges = np.logical_and(ranges > scan_msg.range_min, ranges < scan_msg.range_max)
        valid_angles = angles[valid_ranges]
        valid_ranges = ranges[valid_ranges]
        self.ax.clear()
        self.ax.scatter(valid_angles, valid_ranges, s=1)
        self.ax.set_theta_zero_location('N')
        self.ax.set_title('Laser Scan Data')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.get_logger().info('Received lidar data')
        # self.get_logger().info(f'Ranges: x={scan_msg.ranges}')

def main():
    rclpy.init()
    parse_lidar = ParseLidar()
    rclpy.spin(parse_lidar)
    parse_lidar.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
