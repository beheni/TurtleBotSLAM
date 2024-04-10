#/bin/bash

colcon build --symlink-install --packages-select turtleSLAM
. install/local_setup.bash
