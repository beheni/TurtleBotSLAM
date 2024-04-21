# TurtleBotSLAM
For us, because we keep forgetting

All paths are relative from root of this repo
## Run package
```
source src/turtleSLAM/install/setup.bash
ros2 run turtleSLAM parse_lidar
```
## Run rosbag
```
ros2 bag play -r 1 -l scan_data_bag/scan_data.db3
```
Totally not sure about the rate. `-l` == `--loop`
