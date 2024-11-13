#!/bin/bash
source /opt/ros/noetic/setup.bash

# Build the project and run tests
rm -rf build
mkdir -p build
cd build

if cmake .. -DBUILD_TESTING=ON; then
    echo "CMake successful."
    if make test_depth_anything_v2; then
        echo "Make successful."
    else
        echo "Make failed."
        exit 1
    fi
else
    echo "CMake failed."
    exit 1
fi

# Run the test executable
if ./devel/lib/usb_cam/test_depth_anything_v2; then
    echo "Tests successful."
else
    echo "Tests failed."
    exit 1
fi
