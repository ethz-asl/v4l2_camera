#include "usb_cam/learning/depth_anything_v2.hpp"
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <gtest/gtest.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

// This class provides access to protected members that we normally don't want to expose
class DepthAnythingV2Test : public DepthAnythingV2 {
    public:
        DepthAnythingV2Test(const std::string& model_path) : DepthAnythingV2(nullptr, model_path) {}
        float* get_input_data() { return _input_data; }
};

class TestDepthAnythingV2 : public ::testing::Test {
protected:
    const std::string model_path = "/workspaces/v4l2_camera/test/resources/depth_anything_v2_vits_16.onnx";
    const std::string test_image_path = "/workspaces/v4l2_camera/test/resources/maschinenhalle_example_frame.jpg";

    DepthAnythingV2Test* depth_anything_v2;

    void SetUp() override {
        // Instantiate the Raft model with the test parameters
        depth_anything_v2 = new DepthAnythingV2Test(model_path);
        depth_anything_v2->load_model();
    }

    void TearDown() override {
        delete depth_anything_v2;
    }
};

TEST_F(TestDepthAnythingV2, TestModelLoad) {
    ASSERT_NE(depth_anything_v2->get_engine(), nullptr);
    ASSERT_NE(depth_anything_v2->get_context(), nullptr);
    ASSERT_NE(depth_anything_v2->get_runtime(), nullptr);
}

TEST_F(TestDepthAnythingV2, TestSetInput) {
    // Load the image into a sensor_msgs::Image object
    cv::Mat img = cv::imread(test_image_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Test image could not be loaded!";

    sensor_msgs::Image msg;
    cv_bridge::CvImage cv_image;
    cv_image.image = img;
    cv_image.encoding = sensor_msgs::image_encodings::RGB8;
    cv_image.toImageMsg(msg);

    // Call the set_input function with the loaded image
    depth_anything_v2->set_input(msg);
    ASSERT_NE(depth_anything_v2->get_input_data(), nullptr);

    const size_t expected_size = 518 * 518 * 3;
    float* input_data = depth_anything_v2->get_input_data();
    ASSERT_NE(input_data, nullptr);
    ASSERT_FLOAT_EQ(input_data[0], img.at<cv::Vec3b>(0, 0)[0] / 255.0f);
}

TEST_F(TestDepthAnythingV2, TestPredict) {
    for (size_t i = 0; i < 10; i++) {
        depth_anything_v2->predict();
    }
}
