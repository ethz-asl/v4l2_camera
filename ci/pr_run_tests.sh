#!/bin/bash
source /opt/ros/noetic/setup.bash

rm -rf build
mkdir -p build
cd build
if cmake .. -DBUILD_TESTING=ON
then
    echo "CMake successfull"
    if make test_learning_interface
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

if ./devel/lib/usb_cam/test_learning_interface
then
    echo "Tests successful"
else
    echo "Tests failed"
    exit 1
fi


