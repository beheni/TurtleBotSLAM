#include "slam_cpp/coordinate_converter.h"
#include <functional>
#include <memory>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

namespace slam_cpp
{
    CoordinateConverter::CoordinateConverter() : Node("coordinate_converter")
    {
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        last_odom_time_ = now();
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 3), qos_profile);
        RCLCPP_INFO(get_logger(), "Subscribing to /odom");
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>("/odom", qos, std::bind(&CoordinateConverter::odometry_callback, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "Subscribing to /scan");
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>("/scan", qos, std::bind(&CoordinateConverter::scan_callback, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "creating /abs_coordinates publisher");
        timer_ = create_wall_timer(std::chrono::milliseconds(200), std::bind(&CoordinateConverter::coordinate_conversion, this));
        abs_coordinates_pub_ = create_publisher<interfaces::msg::AbsCoordinates>("/abs_coordinates", qos);
        delta_t_ = 0;
        last_pose_ = nullptr;
        last_scan_ = nullptr;
    }

    CoordinateConverter::~CoordinateConverter() = default;

    double CoordinateConverter::euclidean_distance(const geometry_msgs::msg::Pose2D p1, const geometry_msgs::msg::Pose2D p2){
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }

    double CoordinateConverter::angle_difference(const double a1, const double a2){
        return atan2(sin(a1 - a2), cos(a1 - a2));
    }

    void CoordinateConverter::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg){
        // RCLCPP_INFO(get_logger(), "Received odometry message");
        std::lock_guard lock_o(odometry_lock_);
        {
            if (last_pose_) {
                delta_t_ = (rclcpp::Time(msg->header.stamp) - last_odom_time_).seconds();
                last_odom_time_ = rclcpp::Time(msg->header.stamp);
                last_v = msg->twist.twist.linear.x;
                last_w = msg->twist.twist.angular.z;
                geometry_msgs::msg::Pose2D p1 = calculate_new_pose(*last_pose_, msg->twist.twist, delta_t_);
                if (euclidean_distance(*last_pose_, p1) >= DISTANCE_THRESHOLD ||
                    angle_difference(last_pose_->theta, p1.theta) >= ANGLE_THRESHOLD ){
                    last_pose_ = std::make_shared<geometry_msgs::msg::Pose2D>(p1);
                    }
                return;
            }
            RCLCPP_INFO(get_logger(), "First odometry message");
            geometry_msgs::msg::Pose2D p1;
            last_odom_time_ = rclcpp::Time(msg->header.stamp);
            last_v = msg->twist.twist.linear.x;
            last_w = msg->twist.twist.angular.z;
            p1.x = msg->pose.pose.position.x;
            p1.y = msg->pose.pose.position.y;
            p1.theta = tf2::getYaw(msg->pose.pose.orientation);
            last_pose_ = std::make_shared<geometry_msgs::msg::Pose2D>(p1);
        }
    }

    geometry_msgs::msg::Point CoordinateConverter::polar_to_cartesian(const double ro, const double phi) {
        geometry_msgs::msg::Point p;
        p.x = last_pose_->x + ro * cos(phi + last_pose_->theta);
        p.y = last_pose_->y + ro * sin(phi + last_pose_->theta);
        return p;
    }


    void CoordinateConverter::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg){
        // RCLCPP_INFO(get_logger(), "Received scan message");
        std::lock_guard lock_s(scan_lock_);
        if (last_scan_) {
            for (size_t i = 0; i < msg->ranges.size(); i++){
                if (std::fabs(msg->ranges[i] - last_scan_->ranges[i]) >= SCAN_DELTA){
                    last_scan_ = msg;
                    return;
                }
            }
            return;
        }
        RCLCPP_INFO(get_logger(), "First scan message");
        last_scan_ = msg;
    }

    void CoordinateConverter::coordinate_conversion(){
        if (!last_pose_ || !last_scan_){
            RCLCPP_WARN(get_logger(), "No odometry or scan data");
            return;
        }
        // RCLCPP_INFO(get_logger(), "Converting coordinates");
        interfaces::msg::AbsCoordinates abs_coordinates;
        std::lock_guard lock_o(odometry_lock_);
        {
            abs_coordinates.cur_pose = *last_pose_;
            abs_coordinates.delta_t = delta_t_;
            abs_coordinates.v = last_v;
            abs_coordinates.w = last_w;
            std::lock_guard lock_s(scan_lock_);
            {
                for (size_t i = 0; i < last_scan_->ranges.size(); i+=LIDAR_STEP){
                    abs_coordinates.absolute_scan.push_back(polar_to_cartesian(last_scan_->ranges[i], last_scan_->angle_min + i * last_scan_->angle_increment));
                    geometry_msgs::msg::Point point;
                    point.x = last_scan_->ranges[i];
                    point.y = last_scan_->angle_min + i * last_scan_->angle_increment;
                    abs_coordinates.polar_scan.push_back(point);
                }
            }
        }

        abs_coordinates_pub_->publish(abs_coordinates);
        // RCLCPP_INFO(get_logger(), "Published abs_coordinates");
    }

    double CoordinateConverter::normalize_theta(double theta){
        if (theta > M_PI) {
            theta = std::fmod(theta, M_PI);
            theta -= M_PI;
        }
        if (theta < -M_PI) {
            theta = std::fmod(theta, M_PI);
        }
        return theta;
    }

    geometry_msgs::msg::Pose2D CoordinateConverter::calculate_new_pose(const geometry_msgs::msg::Pose2D& p1, const geometry_msgs::msg::Twist& twist, double delta_t){
        geometry_msgs::msg::Pose2D p2;
        if (twist.angular.z == 0) {
            p2.x = p1.x + twist.linear.x * cos(p1.theta) * delta_t;
            p2.y = p1.y + twist.linear.x * sin(p1.theta) * delta_t;
            p2.theta = p1.theta;
        }
        else {
            p2.x = p1.x + (twist.linear.x / twist.angular.z) *
                (sin(p1.theta + twist.angular.z * delta_t) - sin(p1.theta));
            p2.y = p1.y - (twist.linear.x / twist.angular.z) *
                (cos(p1.theta + twist.angular.z * delta_t) - cos(p1.theta));
            p2.theta = normalize_theta(p1.theta + twist.angular.z * delta_t);
        }
        return p2;
    }

} // namespace slam_cpp



