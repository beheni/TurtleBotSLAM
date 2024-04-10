import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy import qos

class ParseOdom(Node):
    def __init__(self):
        # qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        #     history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        #     )
        super().__init__('parse_odom')
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos_profile=qos.qos_profile_sensor_data)
        self.subscription 
    
    def odom_callback(self, msg):
        self.get_logger().info('Received odometry message')
        self.get_logger().info('Position: x=%f, y=%f, z=%f' % (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
        self.get_logger().info('Orientation: x=%f, y=%f, z=%f, w=%f' % (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))


def main():
    rclpy.init()
    parse_odom = ParseOdom()
    rclpy.spin(parse_odom)
    parse_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
