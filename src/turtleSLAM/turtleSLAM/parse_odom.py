import rclpy
import numpy as np
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber



class ParseOdom(Node):
    
    @staticmethod
    def F(coordinates, v, w, dt):
        _, _, theta = coordinates
        if w == 0:
            F = np.array([[1, 0, -v*np.sin(theta)*dt],
                          [0, 1, v*np.cos(theta)*dt],
                          [0, 0, 1]])
        else:
            F = np.array([[1, 0, -v/w*np.cos(theta) + v/w*np.cos(theta + w*dt)],
                      [0, 1, -v/w*np.sin(theta) + v/w*np.sin(theta + w*dt)],
                      [0, 0, 1]])
        return F
    

    def __init__(self):
        super().__init__('parse_odom')
        self.coordinates = np.array([0, 0, 0])
        self.landmarks = None
        self.sigma = self.R = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-6]])
        self.landmarks_covariance = None
        self.delta = 0.25
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, Odometry, '/odom', qos_profile=qos.qos_profile_sensor_data), 
                                             Subscriber(self, LaserScan, '/scan')], 10, self.delta)
        self.subscription.registerCallback(self.callback)
    
    def callback(self, odometry, lidar):
        v = odometry.twist.twist.linear.x
        w = odometry.twist.twist.angular.z
        angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)
        ranges = np.array(lidar.ranges)
        ranges = np.array(list(zip(angles, ranges)))
        interval = ranges.shape[0] // 10
        ranges = ranges[np.where(ranges[:,1] != np.inf)]
        self.prediction_step(v, w, self.delta)
        self.get_logger().info(f'Odometry: v={v:.3f}, w={w:.3f}')
        self.get_logger().info(f'Lidar: {ranges[::interval]}')

    def prediction_step(self, v, w, dt):
        x, y, theta = self.coordinates
        if w == 0:
            x += v*np.cos(theta)*dt
            y += v*np.sin(theta)*dt
        else:
            x = -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
            y = v/w * np.cos(theta) - v/w * np.cos(theta + w*dt)
            theta = w*dt
        F = ParseOdom.F(self.coordinates, v, w, dt)
        self.sigma = F @ self.sigma @ F.T + self.R
        self.coordinates = np.array([x, y, theta])
        self.get_logger().info(f'Coordinates: {self.coordinates}')


def main():
    rclpy.init()
    parse_odom = ParseOdom()
    rclpy.spin(parse_odom)
    parse_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
