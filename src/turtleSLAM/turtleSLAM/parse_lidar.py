import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
from rclpy import qos
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber

from tf2_ros import TransformBroadcaster, TransformStamped

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

def create_static_transformation(transformation):
    t = TransformStamped()
    t.header.stamp = rclpy.clock.Clock().now().to_msg()
    t.header.frame_id = "rplidar_link"
    t.child_frame_id = transformation[1]

    t.transform.translation.x = float(transformation[2])
    t.transform.translation.y = float(transformation[3])
    t.transform.translation.z = float(transformation[4])

    quat = quaternion_from_euler(
            float(transformation[5]), float(transformation[6]), float(transformation[7]))
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t

class Visualize(Node):
    def __init__(self):
        super().__init__('visualize_data')
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, PoseStamped, "/position"),
                                                          Subscriber(self, PointCloud2, "/landmarks"), 
                                                          Subscriber(self, LaserScan, "/scan")], 10, 0.1)
        self.subscription.registerCallback(self.SLAM_callback)
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
    
    def SLAM_callback(self, position, landmarks, scan):
        x,y,theta = position.pose.position.x, position.pose.position.y, np.arccos(position.pose.orientation.z)
    
        self.get_logger().info(f'Position: {x}, {y}, {theta}')
        lm = np.frombuffer(landmarks.data, dtype=np.float64).reshape(-1, 2)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        angles = angles.reshape(2, -1)
        angles = np.concatenate((angles[1], angles[0]))
        angles -= np.pi/2 
        angles = np.array([self.normalize_angle(angle) for angle in angles])

        ranges = np.array(scan.ranges)

        valid_ranges = np.logical_and(ranges > scan.range_min, ranges < scan.range_max)
        angles = angles[valid_ranges]
        valid_ranges = ranges[valid_ranges]
        ranges = np.array(list(zip(valid_ranges, angles)))
        # self.get_logger().info(f'Ranges: {ranges.shape}')
        coords = np.array([x, y, theta])
        transformed_ranges = self.transform_scan(ranges, coords)
        # self.get_logger().info(f'Transformed ranges: {transformed_ranges.shape}')
    
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
        self.ax.scatter(transformed_ranges[:, 0], transformed_ranges[:, 1], c='g', s=3, label='Lidar Scan')
        self.ax.scatter(lm[:, 0], lm[:, 1], c='r', s=4, label='Landmarks')



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
