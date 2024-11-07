#!/bin/bash
source /opt/ros/noetic/setup.bash

# Check if the plan file exists before generating it
echo $LD_LIBRARY_PATH
if [ ! -f "test/resources/raft-small.plan" ]; then
    echo "Plan file not found. Generating plan file..."
    if /usr/src/tensorrt/bin/trtexec --buildOnly --onnx="test/resources/raft-small.onnx" --saveEngine="test/resources/raft-small.plan" --plugins="/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"
    then
        echo "Plan file generation successful"
    else
        echo "Plan file generation failed"
        exit 1
    fi
else
    echo "Plan file already exists. Skipping generation."
fi

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


