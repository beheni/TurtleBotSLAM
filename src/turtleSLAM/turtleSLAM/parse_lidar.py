import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy import qos

class ParseLidar(Node):
    def __init__(self):
        super().__init__('parse_odom')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            qos_profile=qos.qos_profile_sensor_data)
        self.subscription 
    
    def lidar_callback(self, msg):
        self.get_logger().info('Received lidar data')
        self.get_logger().info(f'Ranges: x={msg.ranges}')

def main():
    rclpy.init()
    parse_lidar = ParseLidar()
    rclpy.spin(parse_lidar)
    parse_lidar.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
