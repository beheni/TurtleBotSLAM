import rclpy
import numpy as np
from rclpy import qos
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber



class RunEKF(Node):
    
    @staticmethod
    def H(q_k, delta_k, k, num_landmarks):
        common = 1/q_k
        root_q_k = np.sqrt(q_k)

        first_block1 = common * np.array([-root_q_k*delta_k[0], -root_q_k*delta_k[1], 0])
        first_block2 = common * np.array([delta_k[1], -delta_k[0], -1])
        first_block = np.block([first_block1, first_block2])

        second_block1 = common * np.array([root_q_k*delta_k[0], root_q_k*delta_k[1]])
        second_block2 = common * np.array([-delta_k[1], delta_k[0]])
        second_block = np.block([second_block1, second_block2])

        first_zeroes = np.zeros((2, 2*k))
        second_zeroes = np.zeros((2, 2*(num_landmarks - k - 2)))
        H = np.block([[first_block, first_zeroes], [second_block, second_zeroes]])
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
        super().__init__('parse_odom')
        self.coordinates = np.array([0, 0, 0])
        self.landmarks = [] # Nx2
        self.sigma_xx = self.R = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-6]])
        self.sigma_mm = None
        self.sigma_xm = None
        self.delta = 0.25
        self.subscription = ApproximateTimeSynchronizer([Subscriber(self, Odometry, '/odom', qos_profile=qos.qos_profile_sensor_data), 
                                             Subscriber(self, LaserScan, '/scan')], 10, self.delta)
        self.subscription.registerCallback(self.callback)

        self.Q_lidar_error = np.array([[8e-6, 0], [0, 1e-6]])

        self.threshold_alpha = 0.1
    
    def callback(self, odometry, lidar):
        v = odometry.twist.twist.linear.x
        w = odometry.twist.twist.angular.z
        angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)
        ranges = np.array(lidar.ranges)
        ranges = np.array(list(zip(ranges, angles)))
        interval = ranges.shape[0] // 10
        ranges = ranges[np.where(ranges[:,0] != np.inf)]
        ranges = ranges[::interval]
        self.prediction_step(v, w, self.delta)
        self.correction_step(ranges) #(Nx2)
        
        self.get_logger().info(f'Odometry: v={v:.3f}, w={w:.3f}')
        self.get_logger().info(f'Lidar: {ranges}')

    def prediction_step(self, v, w, dt):
        x, y, theta = self.coordinates
        if w == 0:
            x += v*np.cos(theta)*dt
            y += v*np.sin(theta)*dt
        else:
            x = -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
            y = v/w * np.cos(theta) - v/w * np.cos(theta + w*dt)
            theta = w*dt
        G = RunEKF.G(self.coordinates, v, w, dt)

        self.sigma_xx = G @ self.sigma_xx @ G.T + self.R
        if self.landmarks != []:
            self.sigma_xm = G @ self.sigma_xm
        
        self.coordinates = np.array([x, y, theta])
        self.get_logger().info(f'Coordinates: {self.coordinates}')
    
    def correction_step(self, ranges):
        x, y, theta = self.coordinates
        for ro, phi in ranges:
            self.get_logger().info(f'ro: {ro}, phi: {phi}')
            maybe_new_landmark = np.array([x + ro*np.cos(phi + theta), y + ro*np.sin(phi + theta)])
            distances = np.zeros(len(self.landmarks))
            for k in range(len(self.landmarks)):
                self.get_logger().info(f'Landmark: {self.landmarks[k]}')
                z_k = self.landmarks[k]
                delta_k = z_k - np.array([x, y])
                q_k = delta_k.T @ delta_k
                z_k_hat = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])
                H = RunEKF.H(q_k, delta_k, k, len(self.landmarks))

                sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])
                psi = H @ sigma @ H.T + self.Q_lidar_error
                pi_k_distance = (maybe_new_landmark - z_k_hat).T @ np.linalg.inv(psi) @ (maybe_new_landmark - z_k_hat)
                distances[k] = pi_k_distance
            # distances_curr = self.threshold_alpha
            landmark_index = np.argmin(distances)
            # landmark = self.landmarks[landmark_index]
            if distances[landmark_index] > self.threshold_alpha:
                self.landmarks = np.vstack([self.landmarks, maybe_new_landmark])
                self.sigma_mm = np.block([[self.sigma_mm, np.zeros((2, 2))], [np.zeros((2, 2)), self.Q_lidar_error]])
                self.sigma_xm = np.block([[self.sigma_xm, np.zeros((3, 2))], [np.zeros((2, 3)), np.zeros((2, 2))]])
            sigma = np.block([[self.sigma_xx, self.sigma_xm],[self.sigma_xm.T, self.sigma_mm]])


            assined_landmark = self.landmarks[landmark_index]
            delta_k = assined_landmark - np.array([x, y])
            q_k = delta_k.T @ delta_k
            z_k_hat = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])
            H = RunEKF.H(q_k, delta_k, landmark_index, len(self.landmarks))
            psi_new_landmark = H @ sigma @ H.T + self.Q_lidar_error
            Kalman_gain = sigma @ H.T @ np.linalg.inv(psi_new_landmark)
            Kalman_gain_new_landmark = Kalman_gain @ (maybe_new_landmark - z_k_hat)
            self.coordinates += Kalman_gain_new_landmark[:3]
            self.landmarks += Kalman_gain_new_landmark[3:].reshape(-1, 2) #errorneous behaviour
            sigma_new = (np.eye(3 + 2*len(self.landmarks)) - Kalman_gain @ H) @ sigma 
            self.sigma_xx = sigma_new[:3, :3]
            self.sigma_mm = sigma_new[3:, 3:]
            self.sigma_xm = sigma_new[:3, 3:]
        self.get_logger().info(f'Landmarks: {self.landmarks}')
        self.get_logger().info(f'Robot coords: {self.coordinates}')
        

def main():
    rclpy.init()
    parse_odom = RunEKF()
    rclpy.spin(parse_odom)
    parse_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
