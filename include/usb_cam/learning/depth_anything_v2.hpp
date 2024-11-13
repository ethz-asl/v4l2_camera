#ifndef DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_HPP_

#include "interface.hpp"
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

class DepthAnythingV2 : public LearningInterface {
public:
    DepthAnythingV2(ros::NodeHandle* nh, const std::string& model_path, const std::string& metric_topic) {
        _model_path = model_path;
        _load_model();

        if (nh != nullptr && !metric_topic.empty()) {
            _depth_pub = nh->advertise<sensor_msgs::Image>(metric_topic, 1);
        }
    }

    void set_input(sensor_msgs::Image& msg) override {
        // Keep track of input image size, we want to get the same output image dimensions
        _output_image_w = msg.width;
        _output_image_h = msg.height;

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat image = cv_ptr->image;

        // Change size to 518x518 (still uint8)
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(_input_w, _input_h));

        std::vector<float> input = _preprocess(resized_image);
        memcpy(_input_data, input.data(), _input_size_float);
    }

void publish() override {
    if (_depth_pub.getTopic() != "") {
        cv::Mat depth_prediction(_input_w, _input_h, CV_32FC1, _output_data);
        cv::Mat depth_resized;
        cv::resize(depth_prediction, depth_resized, cv::Size(_output_image_w, _output_image_h));
        cv_bridge::CvImage depth_image;
        depth_image.header.stamp = ros::Time::now();
        depth_image.header.frame_id = "depth_frame";
        depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depth_image.image = depth_resized;
        _depth_pub.publish(depth_image.toImageMsg());
    }
}

private:
    // TODO: Not so nice magic numbers from the paper implementation
    const float mean[3] = { 123.675f, 116.28f, 103.53f };
    const float std[3] = { 58.395f, 57.12f, 57.375f };
    ros::Publisher _depth_pub;
    uint32_t _output_image_w, _output_image_h;

    std::vector<float> _preprocess(cv::Mat& image) {
        // Determine the resized dimensions
        const int iw = image.cols;
        const int ih = image.rows;
        const float aspect_ratio = static_cast<float>(iw) / ih;
        const int nw = (aspect_ratio >= 1) ? _input_w : static_cast<int>(_input_w * aspect_ratio);
        const int nh = (aspect_ratio >= 1) ? static_cast<int>(_input_h / aspect_ratio) : _input_h;

        // Resize image and pad if necessary
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        cv::Mat padded_image = cv::Mat::ones(cv::Size(_input_w, _input_h), CV_8UC3) * 128;
        resized_image.copyTo(padded_image(cv::Rect((_input_w - nw) / 2, (_input_h - nh) / 2, nw, nh)));

        // Normalize and flatten the image to a 1D tensor
        std::vector<float> input_tensor;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < padded_image.rows; i++) {
                for (int j = 0; j < padded_image.cols; j++) {
                    input_tensor.emplace_back((padded_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
                }
            }
        }

        return input_tensor;
    }
};

#endif // DEPTH_ANYTHING_HPP_
