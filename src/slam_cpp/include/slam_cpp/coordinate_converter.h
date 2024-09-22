#ifndef COORDINATE_CONVERTER_H
#define COORDINATE_CONVERTER_H

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <interfaces/msg/abs_coordinates.hpp>


namespace slam_cpp {
    class CoordinateConverter : public rclcpp::Node {
        static constexpr size_t LIDAR_STEP = 18;
        static constexpr double DISTANCE_THRESHOLD = 0.005;
        static constexpr double ANGLE_THRESHOLD = 0.01;

        static constexpr double SCAN_DELTA = 0.005;

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
        rclcpp::Publisher<interfaces::msg::AbsCoordinates>::SharedPtr abs_coordinates_pub_;
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Time last_odom_time_;
        double delta_t_;
        double last_v;
        double last_w;
        mutable std::mutex odometry_lock_;
        mutable std::mutex scan_lock_;


        geometry_msgs::msg::Pose2D::SharedPtr last_pose_;
        sensor_msgs::msg::LaserScan::SharedPtr last_scan_;
        void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
        void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
        void coordinate_conversion();

        static double euclidean_distance(const geometry_msgs::msg::Pose2D p1, const geometry_msgs::msg::Pose2D p2);
        static double angle_difference(const double a1, const double a2);
        static geometry_msgs::msg::Pose2D calculate_new_pose(const geometry_msgs::msg::Pose2D& p1, const geometry_msgs::msg::Twist& twist, double delta_t);

        geometry_msgs::msg::Point polar_to_cartesian(const double r, const double theta);

    public:
        CoordinateConverter();
        ~CoordinateConverter();
        static double normalize_theta(double theta);
    };


} // namespace slam_cpp

#endif //COORDINATE_CONVERTER_H
