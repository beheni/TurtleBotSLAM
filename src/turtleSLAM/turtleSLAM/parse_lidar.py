import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
from rclpy import qos
from interfaces.msg import SLAMdata

class Visualize(Node):
    def __init__(self):
        super().__init__('visualize_data')
        self.subscription = self.create_subscription(
            SLAMdata,
            'SLAM_data',
            self.SLAM_callback,
            qos_profile=qos.qos_profile_sensor_data)
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Enable interactive mode for continuous updating
        plt.show()

    @staticmethod
    def transform_to_rectangular(msg, coords):
        x, y, theta = coords
        ro, phi = msg
        x += ro * np.cos(phi + theta)
        y += ro * np.sin(phi + theta)
        return np.array([x, y])
        

    def transform_scan(self, scan_msg, coords):
        return np.apply_along_axis(self.transform_to_rectangular, 1, scan_msg, coords)
        

    
    def SLAM_callback(self, scan_msg):
        angles = np.linspace(scan_msg.scan_data.angle_min, scan_msg.scan_data.angle_max, len(scan_msg.scan_data.ranges))
        angles = angles.reshape(2, -1)
        angles = np.concatenate((angles[1], angles[0]))
        angles -= np.pi/2
        self.get_logger().info(f'Angles: {angles}')
        ranges = np.array(scan_msg.scan_data.ranges)
        
        coords = np.array([scan_msg.robot_coords.x, scan_msg.robot_coords.y, scan_msg.robot_coords.theta])
        landmarks = np.array(scan_msg.landmarks).reshape(-1, 2)

        valid_ranges = np.logical_and(ranges > scan_msg.scan_data.range_min, ranges < scan_msg.scan_data.range_max)
        angles = angles[valid_ranges]
        valid_ranges = ranges[valid_ranges]
        ranges = np.array(list(zip(valid_ranges, angles)))
        self.get_logger().info(f'Ranges: {ranges.shape}')
        transformed_ranges = self.transform_scan(ranges, coords)
        self.get_logger().info(f'Transformed ranges: {transformed_ranges.shape}')
    
        self.ax.clear()
        self.ax.set_title('SLAM Data')

        self.ax.set_xticks(np.arange(-100, 100, 1))
        self.ax.set_yticks(np.arange(-100, 100, 1))

        dx = 0.5 * np.cos(coords[2])
        dy = 0.5 * np.sin(coords[2])
        self.ax.arrow(coords[0], coords[1], dx, dy, width=0.1, color="blue", label='Robot Path')
        self.ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2, label='Landmarks')
        self.ax.scatter(transformed_ranges[:, 0], transformed_ranges[:, 1], c='g', s=2, label='Lidar Scan')



        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    visualize = Visualize()
    rclpy.spin(visualize)
    visualize.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
