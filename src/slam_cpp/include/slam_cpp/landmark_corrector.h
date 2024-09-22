#ifndef LANDMARK_CORRECTOR_H
#define LANDMARK_CORRECTOR_H

#include <rclcpp/rclcpp.hpp>
#include <interfaces/msg/abs_coordinates.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <interfaces/msg/landmarks.hpp>

namespace slam_cpp {
    class LandmarkCorrector : public rclcpp::Node {
        struct landmark_data {
            Eigen::Vector2d delta;
            double q;
            Eigen::Vector2d polar_hat;
        };
        rclcpp::Subscription<interfaces::msg::AbsCoordinates>::SharedPtr abs_coordinates_sub_;
        Eigen::Vector<double, 3> coordinates_;
        Eigen::Vector<double, Eigen::Dynamic> map_vector_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> map_covariance_;
        Eigen::Matrix<double, 3, 3> coorinates_covariance_;
        Eigen::Matrix<double, 3, Eigen::Dynamic> cross_covariance_;
        double landmark_treshold_;
        Eigen::Matrix <double, 3, 3> R;
        Eigen::Matrix <double, 2, 2> Q;

        rclcpp::Publisher<interfaces::msg::Landmarks>::SharedPtr corrected_landmarks_pub_;

        void abs_coordinates_callback(interfaces::msg::AbsCoordinates::SharedPtr msg);
        void init_R_Q();
        landmark_data calculate_landmark_data(const Eigen::Vector2d& landmark);
        Eigen::Matrix<double, 2, Eigen::Dynamic> H_func(double q_k, Eigen::Vector<double, 2> delta_k, size_t k, size_t n);
        Eigen::Matrix<double, 3, 3> G_func(interfaces::msg::AbsCoordinates::SharedPtr msg);
        void publish_landmarks();
        void prediction_step(interfaces::msg::AbsCoordinates::SharedPtr msg);

    public:
        LandmarkCorrector();
        ~LandmarkCorrector() = default;
    };

} // namespace slam_cpp

#endif //LANDMARK_CORRECTOR_H
