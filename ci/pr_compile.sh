#!/bin/bash
source /opt/ros/noetic/setup.bash

rm -rf build
mkdir -p build
cd build
if cmake ..
then
    echo "CMake successfull"
    if make
    then
        echo "Make successfull"
    else
        echo "Make failed"
        exit 1
    fi
else
    echo "CMake failed"
    exit 1
fi
