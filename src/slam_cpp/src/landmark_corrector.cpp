#include <slam_cpp/landmark_corrector.h>
#include <cmath>
#include <valarray>
#include <slam_cpp/coordinate_converter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace slam_cpp {
    LandmarkCorrector::LandmarkCorrector() : Node("landmark_corrector") {
        landmark_treshold_ = 5000;
        auto qos_profile = rclcpp::QoS(rclcpp::QoSInitialization(rmw_qos_profile_sensor_data.history, 3), rmw_qos_profile_sensor_data);
        abs_coordinates_sub_ = create_subscription<interfaces::msg::AbsCoordinates>("/abs_coordinates",
            qos_profile, std::bind(&LandmarkCorrector::abs_coordinates_callback, this, std::placeholders::_1));

        corrected_landmarks_pub_ = create_publisher<interfaces::msg::Landmarks>("/landmarks", 1);

        coordinates_ = Eigen::Vector<double, Eigen::Dynamic>(3);
        coordinates_ << 0, 0, 0;

        init_R_Q();
        coorinates_covariance_ = R;
    }

    void LandmarkCorrector::init_R_Q() {
        R = Eigen::Matrix<double, 3, 3>::Zero();
        R(0, 0) = 1e-3;
        R(1, 1) = 1e-3;
        R(2, 2) = 1e-6;
        Q = Eigen::Matrix<double, 2, 2>::Zero();
        Q(0, 0) = 8e-6;
        Q(1, 1) = 1e-6;
    }

    Eigen::Matrix<double, 3, 3> LandmarkCorrector::G_func(interfaces::msg::AbsCoordinates::SharedPtr msg) {
        Eigen::Matrix<double, 3, 3> G = Eigen::Matrix<double, 3, 3>::Identity();
        if (msg->w == 0) {
            G(0, 2) = -msg->v * sin(msg->cur_pose.theta);
            G(1, 2) = msg->v * cos(msg->cur_pose.theta);
        }
        else {
            G(0, 2) = -msg->v / msg->w * cos(msg->cur_pose.theta) + msg->v / msg->w * cos(msg->cur_pose.theta + msg->w * msg->delta_t);
            G(1, 2) = -msg->v / msg->w * sin(msg->cur_pose.theta) + msg->v / msg->w * sin(msg->cur_pose.theta + msg->w * msg->delta_t);
        }
        return G;
    }


    void LandmarkCorrector::prediction_step(interfaces::msg::AbsCoordinates::SharedPtr msg) {
        auto G = G_func(msg);
        coorinates_covariance_ = G * coorinates_covariance_ * G.transpose() + R;
        if (map_vector_.size() != 0) {
            cross_covariance_ = G * cross_covariance_;
        }
        coordinates_ << msg->cur_pose.x, msg->cur_pose.y, msg->cur_pose.theta;
    }

    void LandmarkCorrector::abs_coordinates_callback(interfaces::msg::AbsCoordinates::SharedPtr msg) {
        if (coordinates_ == Eigen::Vector<double, Eigen::Dynamic>::Zero(3)) {
            coordinates_ << msg->cur_pose.x, msg->cur_pose.y, msg->cur_pose.theta;
        }
        prediction_step(msg);
        auto abs_coordinates = msg->absolute_scan;
        auto polar_coordinates = msg->polar_scan;
        for (size_t i = 0; i < abs_coordinates.size(); i++) {
            if (polar_coordinates[i].x == std::numeric_limits<double>::infinity() ||
                polar_coordinates[i].x == -std::numeric_limits<double>::infinity() ||
                polar_coordinates[i].x == 0.0) {
                RCLCPP_WARN(get_logger(), "Incorrect polar coordinates, skipping scan %lu", i);
                continue;
            }
            Eigen::Vector<double, 2> new_landmark, assigned_landmark;
            new_landmark << abs_coordinates[i].x, abs_coordinates[i].y;
            Eigen::Vector<double, 2> new_landmark_polar = Eigen::Vector<double, 2>::Zero();
            new_landmark_polar << polar_coordinates[i].x, polar_coordinates[i].y;
            // RCLCPP_INFO(get_logger(), "Processing scan %lu with polar values %.2f %.2f", i, new_landmark_polar(0), new_landmark_polar(1));
            size_t landmark_index = -1;
            if (map_vector_.size() == 0) {
                map_vector_ = new_landmark;
                // RCLCPP_INFO(get_logger(), "No landmarks in map, adding new landmark %.2f %.2f", new_landmark(0), new_landmark(1));
                map_covariance_ = Q;
                cross_covariance_ = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, 2);
                landmark_index = 0;
                assigned_landmark = new_landmark;
            }
            else {
                std::vector<double> distances;\
                Eigen::MatrixXd sigma(3 + map_vector_.size(), 3 + map_vector_.size());
                sigma << coorinates_covariance_, cross_covariance_,
                         cross_covariance_.transpose(), map_covariance_;
                for (size_t j = 0; j < map_vector_.size(); j +=2) {
                    Eigen::Vector2d landmark = map_vector_.block(j, 0, 2, 1);
                    auto [delta, q, polar_hat] = calculate_landmark_data(landmark);
                    auto H = H_func(q, delta, j, map_vector_.size());
                    Eigen::Vector2d difference = new_landmark_polar - polar_hat;
                    difference(1) = CoordinateConverter::normalize_theta(difference(1));
                    Eigen::Matrix2d psi_landmark = H * sigma * H.transpose() + Q;
                    distances.push_back(difference.transpose() * psi_landmark.inverse() * difference);
                }
                auto min_distance = std::min_element(distances.begin(), distances.end());
                // RCLCPP_INFO(get_logger(), "Min distance: %.2f", *min_distance);
                landmark_index = std::distance(distances.begin(), min_distance) * 2;
                if (*min_distance < landmark_treshold_) {
                    // RCLCPP_INFO(get_logger(), "Assigning new landmark to existing landmark %lu", landmark_index);
                    assigned_landmark = map_vector_.block(landmark_index, 0, 2, 1);
                }
                else {
                    // RCLCPP_INFO(get_logger(), "Adding new landmark to map");
                    map_vector_.conservativeResize(map_vector_.size() + 2);
                    map_vector_.block(map_vector_.size() - 2, 0, 2, 1) = new_landmark;
                    auto old_covariance = map_covariance_;
                    map_covariance_.resize(map_vector_.size(), map_vector_.size());
                    Eigen::Matrix2Xd zero = Eigen::Matrix<double, 2, Eigen::Dynamic>::Zero(2, map_vector_.size() - 2);
                    map_covariance_ << old_covariance, zero.transpose(), zero, Q;
                    cross_covariance_.conservativeResize(3, map_vector_.size());
                    cross_covariance_.block(0, map_vector_.size() - 2, 3, 2) = Eigen::Matrix<double, 3, 2>::Zero();
                    landmark_index = map_vector_.size() - 2;
                    assigned_landmark = new_landmark;
                }
            }
            Eigen::MatrixXd sigma(3 + map_vector_.size(), 3 + map_vector_.size());
            sigma << coorinates_covariance_, cross_covariance_,
                cross_covariance_.transpose(), map_covariance_;
            auto [delta_k, q_k, polar_hat] = calculate_landmark_data(assigned_landmark);
            Eigen::Vector2d difference = new_landmark_polar - polar_hat;
            difference(1) = CoordinateConverter::normalize_theta(difference(1));
            auto H = H_func(q_k, delta_k, landmark_index, map_vector_.size());
            Eigen::Matrix2d psi_landmark = H * sigma * H.transpose() + Q;
            Eigen::MatrixX2d kalman = sigma * H.transpose() * psi_landmark.inverse();
            Eigen::VectorXd new_state = Eigen::VectorX<double>(3 + map_vector_.size());
            new_state = kalman * difference;
            coordinates_ += new_state.head(3);
            coordinates_(2) = CoordinateConverter::normalize_theta(coordinates_(2));
            auto identity = Eigen::MatrixXd::Identity(3 + map_vector_.size(), 3 + map_vector_.size());
            map_vector_ += new_state.tail(map_vector_.size());
            sigma = (identity - kalman * H) * sigma;
            coorinates_covariance_ = sigma.block(0, 0, 3, 3);
            map_covariance_ = sigma.block(3, 3, map_vector_.size(), map_vector_.size());
            cross_covariance_ = sigma.block(0, 3, 3, map_vector_.size());
        }
        publish_landmarks();
    }

    void LandmarkCorrector::publish_landmarks() {
        interfaces::msg::Landmarks landmarks_msg;
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = now();
        pose.header.frame_id = "map";
        pose.pose.position.x = coordinates_(0);
        pose.pose.position.y = coordinates_(1);
        pose.pose.position.z = coordinates_(2);
        landmarks_msg.pose = pose;
        for (size_t i = 0; i < map_vector_.size(); i += 2) {
            geometry_msgs::msg::Point point;
            point.x = map_vector_(i);
            point.y = map_vector_(i + 1);
            point.z = 0;
            landmarks_msg.landmarks.push_back(point);
        }
        corrected_landmarks_pub_->publish(landmarks_msg);
    }



    Eigen::Matrix<double, 2, Eigen::Dynamic> LandmarkCorrector::H_func(double q_k, Eigen::Vector<double, 2> delta_k, size_t k, size_t n) {
        double common_factor = 1 / q_k;
        double root_q_k = sqrt(q_k);
        Eigen::Matrix<double, 2, 3> block1 = Eigen::Matrix<double, 2, 3>::Zero();
        block1 << -root_q_k * delta_k(0), -root_q_k * delta_k(1), 0 , delta_k(1), -delta_k(0), -q_k;
        Eigen::Matrix<double, 2, 2> block2 = Eigen::Matrix<double, 2, 2>::Zero();
        block2 << root_q_k * delta_k(0), root_q_k * delta_k(1), -delta_k(1), delta_k(0);

        Eigen::MatrixXd zero1 = Eigen::MatrixXd::Zero(2, k);
        Eigen::MatrixXd zero2 = Eigen::MatrixXd::Zero(2, n-k-2);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, n + 3);
        H << block1, zero1, block2, zero2;
        return H * common_factor;

    }

    LandmarkCorrector::landmark_data LandmarkCorrector::calculate_landmark_data(const Eigen::Vector2d &landmark) {
        Eigen::Vector2d delta = landmark - coordinates_.head(2);
        double q = delta.dot(delta);
        Eigen::Vector2d polar_hat;
        polar_hat << sqrt(q), atan2(delta(1, 0), delta(0, 0)) - coordinates_(2);
        polar_hat(1) = CoordinateConverter::normalize_theta(polar_hat(1));
        return {delta, q, polar_hat};
    }


} // namespace slam_cpp


int main(int argc, char *argv[]){
    rclcpp::init(argc, argv);
    spin(std::make_shared<slam_cpp::LandmarkCorrector>());
    rclcpp::shutdown();
    return 0;
}