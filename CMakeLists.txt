cmake_minimum_required(VERSION 3.5)
project(usb_cam)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find ROS and catkin dependencies
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  camera_info_manager
  image_transport
  rosconsole
)

# Find OpenCV, Eigen, and CUDA
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Set CUDA toolkit root directory if not already defined
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
endif()
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
find_package(CUDA REQUIRED)

# Find FFmpeg libraries with PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(avcodec REQUIRED libavcodec)
pkg_check_modules(avutil REQUIRED libavutil)
pkg_check_modules(swscale REQUIRED libswscale)

# Output messages if libraries are found
if(avcodec_FOUND)
  message(STATUS "Found libavcodec: ${avcodec_LIBRARIES}")
endif()

if(avutil_FOUND)
  message(STATUS "Found libavutil: ${avutil_LIBRARIES}")
endif()

if(swscale_FOUND)
  message(STATUS "Found libswscale: ${swscale_LIBRARIES}")
endif()

# Find TensorRT libraries
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu CACHE PATH "Path to TensorRT libraries")
find_library(NVINFER nvinfer PATHS ${TENSORRT_LIB_DIR})
find_library(NVINFER_PLUGIN nvinfer_plugin PATHS ${TENSORRT_LIB_DIR})
find_library(NVONNXPARSER nvonnxparser PATHS ${TENSORRT_LIB_DIR})

message(STATUS "TensorRT NVINFER library found at: ${NVINFER}")
message(STATUS "TensorRT NVINFER_PLUGIN library found at: ${NVINFER_PLUGIN}")
message(STATUS "TensorRT NVONNXPARSER library found at: ${NVONNXPARSER}")

# Check if TensorRT libraries are found
if(NOT NVINFER OR NOT NVINFER_PLUGIN OR NOT NVONNXPARSER)
  message(FATAL_ERROR "TensorRT libraries not found. Ensure TENSORRT_LIB_DIR is set correctly.")
endif()

## Build the USB camera library
add_library(${PROJECT_NAME} SHARED
  src/usb_cam.cpp
  src/interface.cpp
)

# Set include directories
target_include_directories(${PROJECT_NAME} PUBLIC
  "include"
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Link libraries for the usb_cam library
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
  PRIVATE
    ${avcodec_LIBRARIES}
    ${avutil_LIBRARIES}
    ${swscale_LIBRARIES}
    ${NVINFER}
    ${NVINFER_PLUGIN}
    ${NVONNXPARSER}
)

# Define catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# Build the USB camera node executable
add_executable(${PROJECT_NAME}_node src/usb_cam_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${avcodec_LIBRARIES}
    ${avutil_LIBRARIES}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVINFER_PLUGIN}
    ${NVINFER}
    ${NVONNXPARSER}
    ${swscale_LIBRARIES}
)

add_executable(image_processing_node src/learning_node.cpp)
target_link_libraries(image_processing_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Ensure include directories are set
target_include_directories(${PROJECT_NAME}_node PUBLIC
  ${catkin_INCLUDE_DIRS})

# Testing
if(BUILD_TESTING)
  # Find GTest package
  set(CMAKE_BUILD_TYPE Debug)
  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  catkin_add_gtest(test_depth_anything_v2 test/test_depth_anything_v2.cpp)
  target_link_libraries(test_depth_anything_v2
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      GTest::gtest_main
      ${NVINFER}
      ${NVINFER_PLUGIN}
      ${NVONNXPARSER}
      ${CUDA_LIBRARIES}
  )
  set_target_properties(test_depth_anything_v2 PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
endif()

# Installation rules
if($ENV{ROS_VERSION} EQUAL 1)
  install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )

  # Copy launch files
  install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    FILES_MATCHING PATTERN "*.launch"
  )

  # Include header files
  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
else()
  install(PROGRAMS scripts/show_image.py
    DESTINATION lib/${PROJECT_NAME})

  install(TARGETS
    ${PROJECT_NAME}
    ${PROJECT_NAME}_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
  )

  ament_auto_package(
    INSTALL_TO_SHARE
      launch
      config
  )
endif()
