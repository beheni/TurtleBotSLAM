import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from time import time
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class ResetOdm(Node):
    def __init__(self):
        super().__init__('turtle_reset_odm')
        self.subscription = self.create_subscription(
            LaserScan,
            "scan",
            self.listener_callback,
            1000
        )
        self.subscription
    
    def listener_callback(self, msg):
        self.get_logger().info(f"i got \n {msg}")

def main(args=None):
    rclpy.init(args=args)
    node = ResetOdm()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
