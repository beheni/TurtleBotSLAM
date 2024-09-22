import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
from rclpy import qos
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from interfaces.msg import Landmarks
from tf2_ros import TransformBroadcaster, TransformStamped

class Visualize(Node):
    def __init__(self):
        super().__init__('visualize_data')
        self.subscription = self.create_subscription(Landmarks, "/landmarks", self.SLAM_callback, qos.qos_profile_sensor_data)
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
        

    @staticmethod
    def normalize_angle(angle):
        if angle > np.pi:
            angle %= np.pi
            angle -= np.pi
        elif angle < -np.pi:
            angle %= np.pi
        return angle
    
    def SLAM_callback(self, landmarks):
        x,y,theta = landmarks.pose.pose.position.x, landmarks.pose.pose.position.y, landmarks.pose.pose.position.z
    
        self.get_logger().info(f'Position: {x}, {y}, {theta}')
        lm = landmarks.landmarks
        lm_x = [l.x for l in lm]
        lm_y = [l.y for l in lm]

        self.ax.clear()
        self.ax.set_facecolor("#303030")
        self.ax.grid(True)
        self.ax.set_title('SLAM Data')

        self.ax.set_xticks(np.arange(-10, 10, 1))
        self.ax.set_yticks(np.arange(-100, 10, 1))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        self.ax.arrow(x, y, dx, dy, width=0.05, color="white", label='Robot Path')
        self.ax.scatter(lm_x, lm_y, c='r', s=4, label='Landmarks')



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
