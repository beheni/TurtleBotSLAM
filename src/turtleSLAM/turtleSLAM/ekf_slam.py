import rclpy
import numpy as np
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from message_filters import ApproximateTimeSynchronizer, Subscriber

import math
# from std_msgs.msg import String
# from interfaces.msg import SLAMdata

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
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


class RunEKF(Node):
    
    @staticmethod
    def H(q_k, delta_k, k, num_landmarks):
        common = 1/q_k
        root_q_k = np.sqrt(q_k)

        first_block1 = common * np.array([-root_q_k*delta_k[0], -root_q_k*delta_k[1], 0])
        first_block2 = common * np.array([delta_k[1], -delta_k[0], -q_k])
        first_block = np.vstack([first_block1, first_block2])

        second_block1 = common * np.array([root_q_k*delta_k[0], root_q_k*delta_k[1]])
        second_block2 = common * np.array([-delta_k[1], delta_k[0]])
        second_block = np.vstack([second_block1, second_block2])

        first_zeroes = np.zeros((2, 2*k))
        second_zeroes = np.zeros((2, 2*(num_landmarks - k - 1)))
        H = np.block([[first_block, first_zeroes, second_block, second_zeroes]])
        return H
        

    @staticmethod
    def G(coordinates, v, w, dt):
        _, _, theta = coordinates
        if w == 0:
            G = np.array([[1, 0, -v*np.sin(theta)*dt],
                          [0, 1, v*np.cos(theta)*dt],
                          [0, 0, 1]])
        else:   
            G = np.array([[1, 0, -v/w*np.cos(theta) + v/w*np.cos(theta + w*dt)],
                      [0, 1, -v/w*np.sin(theta) + v/w*np.sin(theta + w*dt)],
                      [0, 0, 1]])
        return G
    

    def __init__(self):
        super().__init__('EKF_SLAM')
        transformation = ['rplidar_link', 'base_link', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        static_transform = create_static_transformation(transformation)
        self.coordinates = np.array([0, 0, static_transform.transform.rotation.z])

        self.landmarks =[] # Nx2
        self.sigma_xx = self.R = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-6]])
        self.sigma_mm = []
        self.sigma_xm = []
        self.delta = 0.1
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, Odometry, '/odom', qos_profile=qos.qos_profile_sensor_data), 
                                             Subscriber(self, LaserScan, '/scan')], 10, self.delta)
        self.subscription.registerCallback(self.callback)

        self.Q_lidar_error = np.array([[8e-6, 0], [0, 1e-6]])

        self.threshold_alpha = 0.1
        # self.publisher = self.create_publisher(SLAMdata, '/SLAM_data', qos_profile=qos.qos_profile_sensor_data)
        self.publisher_l = self.create_publisher(PointCloud2, '/landmarks',1)
        self.publisher_p = self.create_publisher(PoseStamped, '/position', 1)
        self.base_x = None
        self.base_y = None


    def callback(self, odometry, lidar):
        v = odometry.twist.twist.linear.x
        w = odometry.twist.twist.angular.z
        if not self.base_x:
            self.base_x = odometry.pose.pose.position.x
            self.base_y = odometry.pose.pose.position.y
        angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)
        angles = angles.reshape(2, -1)
        angles = np.concatenate((angles[1], angles[0]))
        angles -= np.pi/2
        angles = np.array(list(map(self.normalize_angle, angles)))
        ranges = np.array(lidar.ranges)
        interval = 40
        valid_ranges = np.logical_and(ranges > lidar.range_min, ranges < lidar.range_max)
        angles = angles[valid_ranges]
        ranges = ranges[valid_ranges]

        ranges = np.array(list(zip(ranges, angles)))
        # self.get_logger().info(f'angle_min: {lidar.angle_min}\n angle_max: {lidar.angle_max} \nRanges: {ranges}')

        ranges = ranges[::interval]
        self.prediction_step(v, w, self.delta)
        self.correction_step(ranges) #(Nx2)
        self.get_logger().info(f'Coordinates: {self.coordinates}')
        # self.get_logger().info(f'Real coordinates: {odometry.pose.pose.position.x - self.base_x, odometry.pose.pose.position.y - self.base_y}')
        # self.publisher.publish(String(data=str(self.coordinates)))
        robot_pose = PoseStamped()
        robot_pose.pose.position.x =  self.coordinates[0]
        robot_pose.pose.position.y =  self.coordinates[1]
        robot_pose.pose.position.z =  0.0

        robot_pose.header = odometry.header

        robot_pose.pose.orientation.x = 0.0
        robot_pose.pose.orientation.y = 0.0
        robot_pose.pose.orientation.z = np.cos(self.coordinates[2])
        robot_pose.pose.orientation.w = 1.0
        self.publisher_p.publish(robot_pose)


        landmarks = PointCloud2()
        landmarks.header = odometry.header

        landmarks.height = 1
        landmarks.width = self.landmarks.shape[0]
        landmarks.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT64, count=1),
                                        PointField(name='y', offset=8, datatype=PointField.FLOAT64, count=1)]
        landmarks.point_step = 16
        landmarks.row_step = 16*self.landmarks.shape[0]
        landmarks.is_dense = False
        landmarks.is_bigendian = False
        landmarks.data = self.landmarks.flatten().astype(np.float64).tobytes()

        self.publisher_l.publish(landmarks)

        # robot_pose.scan_data = lidar
        # robot_pose.odom_data = odometry
        # robot_pose.landmarks = list(self.landmarks.flatten())
        # 
        
        # self.get_logger().info(f'Coordinates: {self.coordinates}, landmark count: {self.landmarks.shape[0]}')

    @staticmethod
    def normalize_angle(inp):
        def normalize_one(angle):
            if angle > np.pi:
                angle %= np.pi
                angle -= np.pi
            elif angle < -np.pi:
                angle %= np.pi
            return angle
        if isinstance(inp, np.ndarray):
            return np.array([normalize_one(angle) for angle in inp])
        return normalize_one(inp)
    

    def prediction_step(self, v, w, dt):
        x, y, theta = self.coordinates
        if w == 0:
            x += v*np.cos(theta)*dt
            y += v*np.sin(theta)*dt
        else:
            x += -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
            y += v/w * np.cos(theta) - v/w * np.cos(theta + w*dt)
            theta += (w*dt)
            theta = RunEKF.normalize_angle(theta)

        G = RunEKF.G(self.coordinates, v, w, dt)

        self.sigma_xx = G @ self.sigma_xx @ G.T + self.R
        if self.landmarks != []:
            self.sigma_xm = G @ self.sigma_xm
        
        self.coordinates = np.array([x, y, theta])

    @staticmethod
    def choose_landmark(landmark, self):
        x, y, theta = self.coordinates
        delta = landmark - np.array([x, y])
        q = delta.T @ delta
        z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - theta])
        z_hat[1] = RunEKF.normalize_angle(z_hat[1])
        result = np.array([delta[0], delta[1], q, z_hat[0], z_hat[1]])
        return result

    def correction_step(self, ranges):
        x, y, theta = self.coordinates
        for ro, phi in ranges:
            maybe_new_landmark = np.array([x + ro*np.cos(phi + theta), y + ro*np.sin(phi + theta)])
            maybe_new_landmark_polar = np.array([ro, phi])
            assigned_landmark = None
            landmark_index = -1
            if len(self.landmarks) != 0:
                data = np.apply_along_axis(RunEKF.choose_landmark, 1, self.landmarks, self)
                delta = data[:, 0:2]
                q = data[:, 2]
                z_hat = data[:, 3:5]
                H = np.array([RunEKF.H(q[i], delta[i,:], i, len(self.landmarks)) for i in range(len(self.landmarks))])
                sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
                psi = np.matmul(H, sigma) @ H.transpose(0, 2, 1) + self.Q_lidar_error
                difference = - z_hat + maybe_new_landmark_polar
                difference[:, 1] = RunEKF.normalize_angle(difference[:, 1])
                pi_k_distance = np.array([difference[i] @ np.linalg.inv(psi[i]) @ difference[i].T for i in range(len(self.landmarks))]) # rewrite using einsum

                landmark_index = np.argmin(pi_k_distance)
                closest_dist = pi_k_distance[landmark_index]
                if closest_dist > self.threshold_alpha:
                    self.landmarks = np.vstack([self.landmarks, maybe_new_landmark])
                    self.sigma_mm = np.block([[self.sigma_mm, np.zeros((self.sigma_mm.shape[0], 2))],
                                          [np.zeros((2, self.sigma_mm.shape[1])), self.Q_lidar_error]])
                    self.sigma_xm = np.block([[self.sigma_xm, np.zeros((3, 2))]])
                    landmark_index = self.landmarks.shape[0] - 1
                    assigned_landmark = maybe_new_landmark
                else:
                    assigned_landmark = self.landmarks[landmark_index]
            else:
                self.landmarks = np.array([maybe_new_landmark])
                self.sigma_mm = self.Q_lidar_error
                self.sigma_xm = np.zeros((3, 2))
                assigned_landmark = maybe_new_landmark
                landmark_index = 0
            sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
            delta_k = assigned_landmark - np.array([x, y])
            q_k = delta_k.T @ delta_k
            z_k_hat = np.array([np.sqrt(q_k), RunEKF.normalize_angle(np.arctan2(delta_k[1], delta_k[0]) - theta)])

            H = RunEKF.H(q_k, delta_k, landmark_index, len(self.landmarks))
            psi_new_landmark = H @ sigma @ H.T + self.Q_lidar_error
            Kalman_gain = sigma @ H.T @ np.linalg.inv(psi_new_landmark)
            difference = maybe_new_landmark_polar - z_k_hat
            difference[1] = RunEKF.normalize_angle(difference[1])
            Kalman_gain_new_landmark = Kalman_gain @ (difference)

            self.coordinates += Kalman_gain_new_landmark[:3]
            self.coordinates[2] = RunEKF.normalize_angle(self.coordinates[2])
            self.landmarks += Kalman_gain_new_landmark[3:].reshape(-1, 2) #errorneous behaviour
            sigma_new = (np.eye(3 + 2*len(self.landmarks)) - Kalman_gain @ H) @ sigma 
            self.sigma_xx = sigma_new[:3, :3]
            self.sigma_mm = sigma_new[3:, 3:]
            self.sigma_xm = sigma_new[:3, 3:]
            
            

        
    
    # def correction_step(self, ranges):
    #     x, y, theta = self.coordinates
    #     for ro, phi in ranges:
    #         maybe_new_landmark = np.array([x + ro*np.cos(phi + theta), y + ro*np.sin(phi + theta)]) #absolute coordinates
    #         maybe_new_landmark_polar = np.array([ro, phi])
    #         distances = np.zeros(len(self.landmarks))
    #         for k in range(len(self.landmarks)):
    #             z_k = self.landmarks[k] #self landmarks should be in absolute coordinates
    #             delta_k = z_k - np.array([x, y])
    #             q_k = delta_k.T @ delta_k
    #             z_k_hat = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta]) #polar coordinates
    #             z_k_hat[1] = RunEKF.normalize_angle(z_k_hat[1])
    #             H = RunEKF.H(q_k, delta_k, k, len(self.landmarks))    
    #             sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
    #             # self.get_logger().info(f'sigma: {sigma}')
    #             psi = H @ sigma @ H.T + self.Q_lidar_error
    #             # self.get_logger().info(f'psi: {psi}')
    #             # self.get_logger().info(f'psi_inv: {np.linalg.inv(psi)}')
    #             # self.get_logger().info(f'H: {H}')
    #              #relative????
    #             difference = maybe_new_landmark_polar - z_k_hat
    #             difference[1] = RunEKF.normalize_angle(difference[1])
    #             pi_k_distance = difference.T @ np.linalg.inv(psi) @ difference


    #             # self.get_logger().info(f'new_landmark-z_k_hat: {maybe_new_landmark_polar-z_k_hat}')
    #             # self.get_logger().info(f'pi_k_distance: {pi_k_distance}')
    #             # pi_k_distance = (maybe_new_landmark - z_k_hat).T @ psi @ (maybe_new_landmark - z_k_hat)
    #             distances[k] = pi_k_distance

    #         landmark_index = 0
    #         if len(distances) == 0:
    #             self.landmarks = np.array([maybe_new_landmark])
    #             self.sigma_mm = self.Q_lidar_error
    #             self.sigma_xm = np.zeros((3, 2))
    #             landmark_index = 0
    #         elif distances[np.argmin(distances)] > self.threshold_alpha:
    #             self.landmarks = np.vstack([self.landmarks, maybe_new_landmark])
    #             self.sigma_mm = np.block([[self.sigma_mm, np.zeros((self.sigma_mm.shape[0], 2))],
    #                                       [np.zeros((2, self.sigma_mm.shape[1])), self.Q_lidar_error]])
    #             self.sigma_xm = np.block([[self.sigma_xm, np.zeros((3, 2))]])
    #             landmark_index = self.landmarks.shape[0] - 1
    #         else:
    #             landmark_index = np.argmin(distances)

    #         sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
    #         # self.get_logger().info(f'sigma: {sigma}')

    #         assined_landmark = self.landmarks[landmark_index]

    #         delta_k = assined_landmark - np.array([x, y])
    #         q_k = delta_k.T @ delta_k
    #         z_k_hat = np.array([np.sqrt(q_k), RunEKF.normalize_angle(np.arctan2(delta_k[1], delta_k[0]) - theta)])

    #         H = RunEKF.H(q_k, delta_k, landmark_index, len(self.landmarks))
    #         psi_new_landmark = H @ sigma @ H.T + self.Q_lidar_error
    #         Kalman_gain = sigma @ H.T @ np.linalg.inv(psi_new_landmark)
    #         # Kalman_gain = sigma @ H.T @ psi_new_landmark
    #         difference = maybe_new_landmark_polar - z_k_hat
    #         difference[1] = RunEKF.normalize_angle(difference[1])
    #         Kalman_gain_new_landmark = Kalman_gain @ (difference)

    #         self.coordinates += Kalman_gain_new_landmark[:3]
    #         self.coordinates[2] = RunEKF.normalize_angle(self.coordinates[2])
    #         self.landmarks += Kalman_gain_new_landmark[3:].reshape(-1, 2) #errorneous behaviour
    #         sigma_new = (np.eye(3 + 2*len(self.landmarks)) - Kalman_gain @ H) @ sigma 
    #         self.sigma_xx = sigma_new[:3, :3]
    #         self.sigma_mm = sigma_new[3:, 3:]
    #         self.sigma_xm = sigma_new[:3, 3:]


        

def main():
    rclpy.init()
    run_ekf = RunEKF()
    rclpy.spin(run_ekf)
    run_ekf.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
