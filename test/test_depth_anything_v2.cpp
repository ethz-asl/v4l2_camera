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
        DepthAnythingV2Test(const std::string& model_path) : DepthAnythingV2(nullptr, model_path, std::string("")) {}
        float* get_input_data() { return _input_data; }
        void load_model() { _load_model(); }
};

class TestDepthAnythingV2 : public ::testing::Test {
protected:
    const std::string model_path = "../test/resources/depth_anything_v2_vits.onnx";
    const std::string test_image_path = "../test/resources/maschinenhalle_example_frame.jpg";

    cv_bridge::CvImage cv_image;
    cv::Mat img;
    DepthAnythingV2Test* depth_anything_v2;
    sensor_msgs::Image msg;

    void SetUp() override {
        img = cv::imread(test_image_path, cv::IMREAD_COLOR);
        cv_image.image = img;
        cv_image.encoding = sensor_msgs::image_encodings::RGB8;
        cv_image.toImageMsg(msg);

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
    depth_anything_v2->set_input(msg);
    ASSERT_NE(depth_anything_v2->get_input_data(), nullptr);
}

TEST_F(TestDepthAnythingV2, TestPredictPublish) {
    for (size_t i = 0; i < 10; i++) {
        depth_anything_v2->set_input(msg);
        depth_anything_v2->predict();
        depth_anything_v2->publish();
    }
}
