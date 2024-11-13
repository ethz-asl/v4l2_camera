#!/bin/bash
source /opt/ros/noetic/setup.bash

# Set paths for the model and plan files
MODEL_PATH="test/resources/depth_anything_v2_vits.onnx"

# Check if the ONNX model file exists
if [ ! -f "$MODEL_PATH" ]; then
    # If the model file doesn't exist, check if the environment variable is set
    if [ -z "$DEPTH_ANYTHING_V2_VITS_16_LINK" ]; then
        echo "The model file does not exist, and the environment variable DEPTH_ANYTHING_V2_VITS_16_LINK is not set."
        exit 1
    else
        # If the environment variable is set, download the model
        echo "ONNX model file not found. Attempting to download..."
        
        # Create the directory if it doesn't exist
        mkdir -p "$(dirname "$MODEL_PATH")"
        
        # Download the file
        if wget -O "$MODEL_PATH" "$DEPTH_ANYTHING_V2_VITS_16_LINK"; then
            echo "Download successful."
        else
            echo "Download failed."
            exit 1
        fi
    fi
else
    echo "ONNX model file already exists."
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
