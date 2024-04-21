# TurtleBotSLAM
For us, because we keep forgetting
## Run package
```
source install/setup.bash
ros2 run turtleSLAM parse_lidar
```
## Run rosbag
```
ros2 bag play -r 1 -l scan_data.db3
```
Totally not sure about the rate. `-l` == `--loop`
