#/bin/bash

colcon build --symlink-install --packages-select turtleSLAM
source install/local_setup.bash