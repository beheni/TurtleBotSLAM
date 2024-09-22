#include <slam_cpp/coordinate_converter.h>

int main(int argc, char *argv[]){
    rclcpp::init(argc, argv);
    spin(std::make_shared<slam_cpp::CoordinateConverter>());
    rclcpp::shutdown();
    return 0;
}

