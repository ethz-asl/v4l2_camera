#include <gtest/gtest.h>
#include "usb_cam/learning/depth_anything_v2.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <fstream>

// Define a fixture for Raft tests
class DepthAnythingV2Test : public ::testing::Test {
protected:
    // Initialize variables for the test
    std::string model_path = "/workspaces/v4l2_camera/test/resources/depth_anything_v2.plan";

    DepthAnythingV2* depth_anything_v2;

    void SetUp() override {
        // Instantiate the Raft model with the test parameters
        depth_anything_v2 = new DepthAnythingV2(model_path);
        depth_anything_v2->load_model();
    }

    void TearDown() override {
        delete depth_anything_v2;
    }
};

TEST_F(DepthAnythingV2Test, TestModelLoad) {
    ASSERT_NE(depth_anything_v2->get_engine(), nullptr);
    ASSERT_NE(depth_anything_v2->get_context(), nullptr);
    ASSERT_NE(depth_anything_v2->get_runtime(), nullptr);
}
