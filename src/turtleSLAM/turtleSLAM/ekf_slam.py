import rclpy
import numpy as np
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from message_filters import ApproximateTimeSynchronizer, Subscriber

from std_msgs.msg import String
from interfaces.msg import SLAMdata

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
        self.coordinates = np.array([0, 0, 0])
        self.landmarks = [] # Nx2
        self.sigma_xx = self.R = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-6]])
        self.sigma_mm = []
        self.sigma_xm = []
        self.delta = 0.1
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, Odometry, '/odom', qos_profile=qos.qos_profile_sensor_data), 
                                             Subscriber(self, LaserScan, '/scan')], 10, self.delta)
        self.subscription.registerCallback(self.callback)

        self.Q_lidar_error = np.array([[8e-6, 0], [0, 1e-6]])

        self.threshold_alpha = 0.25
        self.publisher = self.create_publisher(SLAMdata, '/SLAM_data', qos_profile=qos.qos_profile_sensor_data)
        self.base_x = None
        self.base_y = None


    def callback(self, odometry, lidar):
        v = odometry.twist.twist.linear.x
        w = odometry.twist.twist.angular.z
        if not self.base_x:
            self.base_x = odometry.pose.pose.position.x
            self.base_y = odometry.pose.pose.position.y
        angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)
        ranges = np.array(lidar.ranges)
        ranges = np.array(list(zip(ranges, angles)))
        interval = 80
        ranges = ranges[np.where(ranges[:,0] != np.inf)]
        ranges = ranges[::interval]
        self.prediction_step(v, w, self.delta)
        self.correction_step(ranges) #(Nx2)
        self.get_logger().info(f'Coordinates: {self.coordinates}')
        # self.get_logger().info(f'Real coordinates: {odometry.pose.pose.position.x - self.base_x, odometry.pose.pose.position.y - self.base_y}')
        # self.publisher.publish(String(data=str(self.coordinates)))
        all_data = SLAMdata()
        all_data.robot_coords = Pose2D(x=self.coordinates[0], y=self.coordinates[1], theta=self.coordinates[2])
        all_data.scan_data = lidar
        all_data.odom_data = odometry
        all_data.landmarks = list(self.landmarks.flatten())

        self.publisher.publish(all_data)
        
        # self.get_logger().info(f'Coordinates: {self.coordinates}, landmark count: {self.landmarks.shape[0]}')

    

    def prediction_step(self, v, w, dt):
        x, y, theta = self.coordinates
        if w == 0:
            x += v*np.cos(theta)*dt
            y += v*np.sin(theta)*dt
        else:
            x += -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
            y += v/w * np.cos(theta) - v/w * np.cos(theta + w*dt)
            theta += (w*dt)
            if theta > np.pi:
                theta %= np.pi
                theta -= np.pi
            elif theta < -np.pi:
                theta %= np.pi

        G = RunEKF.G(self.coordinates, v, w, dt)

        self.sigma_xx = G @ self.sigma_xx @ G.T + self.R
        if self.landmarks != []:
            self.sigma_xm = G @ self.sigma_xm
        
        self.coordinates = np.array([x, y, theta])
        
    
    def correction_step(self, ranges):
        x, y, theta = self.coordinates
        for ro, phi in ranges:
            maybe_new_landmark = np.array([x + ro*np.cos(phi + theta), y + ro*np.sin(phi + theta)]) #absolute coordinates

            distances = np.zeros(len(self.landmarks))
            for k in range(len(self.landmarks)):
                z_k = self.landmarks[k] #self landmarks should be in absolute coordinates


                delta_k = z_k - np.array([x, y])
                q_k = delta_k.T @ delta_k
                z_k_hat = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta]) #polar coordinates

                H = RunEKF.H(q_k, delta_k, k, len(self.landmarks))    
                sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
                # self.get_logger().info(f'sigma: {sigma}')
                psi = H @ sigma @ H.T + self.Q_lidar_error
                # self.get_logger().info(f'psi: {psi}')
                # self.get_logger().info(f'psi_inv: {np.linalg.inv(psi)}')
                # self.get_logger().info(f'H: {H}')
                maybe_new_landmark_polar = np.array([ro, phi]) #relative????

                pi_k_distance = (maybe_new_landmark_polar - z_k_hat).T @ np.linalg.inv(psi) @ (maybe_new_landmark_polar - z_k_hat)


                # self.get_logger().info(f'new_landmark-z_k_hat: {maybe_new_landmark_polar-z_k_hat}')
                # self.get_logger().info(f'pi_k_distance: {pi_k_distance}')
                # pi_k_distance = (maybe_new_landmark - z_k_hat).T @ psi @ (maybe_new_landmark - z_k_hat)
                distances[k] = pi_k_distance


            maybe_new_landmark_polar = np.array([ro, phi])

            landmark_index = 0
            if len(distances) == 0:
                self.landmarks = np.array([maybe_new_landmark])
                self.sigma_mm = self.Q_lidar_error
                self.sigma_xm = np.zeros((3, 2))
                landmark_index = 0
            elif distances[np.argmin(distances)] > self.threshold_alpha:
                self.landmarks = np.vstack([self.landmarks, maybe_new_landmark])
                self.sigma_mm = np.block([[self.sigma_mm, np.zeros((self.sigma_mm.shape[0], 2))],
                                          [np.zeros((2, self.sigma_mm.shape[1])), self.Q_lidar_error]])
                self.sigma_xm = np.block([[self.sigma_xm, np.zeros((3, 2))]])
                landmark_index = self.landmarks.shape[0] - 1
            else:
                landmark_index = np.argmin(distances)

            sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
            # self.get_logger().info(f'sigma: {sigma}')

            assined_landmark = self.landmarks[landmark_index]

            delta_k = assined_landmark - np.array([x, y])
            q_k = delta_k.T @ delta_k
            z_k_hat = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])

            H = RunEKF.H(q_k, delta_k, landmark_index, len(self.landmarks))
            psi_new_landmark = H @ sigma @ H.T + self.Q_lidar_error
            Kalman_gain = sigma @ H.T @ np.linalg.inv(psi_new_landmark)
            # Kalman_gain = sigma @ H.T @ psi_new_landmark
            Kalman_gain_new_landmark = Kalman_gain @ (maybe_new_landmark_polar - z_k_hat)

            self.coordinates += Kalman_gain_new_landmark[:3]
            self.landmarks += Kalman_gain_new_landmark[3:].reshape(-1, 2) #errorneous behaviour
            sigma_new = (np.eye(3 + 2*len(self.landmarks)) - Kalman_gain @ H) @ sigma 
            self.sigma_xx = sigma_new[:3, :3]
            self.sigma_mm = sigma_new[3:, 3:]
            self.sigma_xm = sigma_new[:3, 3:]


        

def main():
    rclpy.init()
    run_ekf = RunEKF()
    rclpy.spin(run_ekf)
    run_ekf.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
