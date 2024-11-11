#ifndef DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_HPP_

#include "interface.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

class DepthAnythingV2 : public LearningInterface {
public:
    DepthAnythingV2(std::string model_path) {
        _model_path = model_path;
        _INPUT_SIZE = cv::Size(_HEIGHT, _WIDTH);
    }

    void set_input(sensor_msgs::Image& msg) override {
        // From ROS msg image to cv mat
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat image = cv_ptr->image;

        // Change size to 518x518 (still uint8)
        cv::Mat resized_image;
        cv::resize(image, resized_image, _INPUT_SIZE);

        // Change to float32 between 0 and 1
        std::vector<cv::Mat> channels;
        cv::split(resized_image, channels);
        for (uint8_t i = 0; i < 3; ++i) {
            channels[i].convertTo(channels[i], CV_32F, 1.0f / 255.0f);
        }
        cv::Mat float_image;
        cv::merge(channels, float_image);
        _input_data = float_image.reshape(1, 1).ptr<float>(0);
    }

    void get_output(uint8_t* output_buffer) override {
        // TODO
    }

    void publish() override {
        // TODO
    }

private:
    const size_t _HEIGHT = 518;
    const size_t _WIDTH = 518;
    cv::Size _INPUT_SIZE;
};

#endif // DEPTH_ANYTHING_HPP_
