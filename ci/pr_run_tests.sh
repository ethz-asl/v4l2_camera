#!/bin/bash
source /opt/ros/noetic/setup.bash

# Set paths for the model and plan files
MODEL_PATH="test/resources/depth_anything_v2_vitb.onnx"
MODEL_URL="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitb.onnx"

# Step 1: Check if the ONNX model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ONNX model file not found. Downloading..."
    if wget -O "$MODEL_PATH" "$MODEL_URL"; then
        echo "Model downloaded successfully."
    else
        echo "Model download failed."
        exit 1
    fi
else
    echo "ONNX model file already exists. Skipping download."
fi

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
