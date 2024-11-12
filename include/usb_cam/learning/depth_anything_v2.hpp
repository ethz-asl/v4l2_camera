#ifndef DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_HPP_

#include "interface.hpp"
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

class DepthAnythingV2 : public LearningInterface {
public:
    DepthAnythingV2(ros::NodeHandle* nh, std::string& model_path, std::string& metric_topic, std::string& vis_topic) {
        _INPUT_SIZE = cv::Size(_HEIGHT, _WIDTH);
        _model_path = model_path;
        _load_model();

        if (nh != nullptr) {
            if (!metric_topic.empty()) {
                _depth_metric_pub = nh->advertise<sensor_msgs::Image>(metric_topic, 1);
            }
            if (!vis_topic.empty()) {
                _depth_vis_pub = nh->advertise<sensor_msgs::Image>(vis_topic, 1);
            }
            
        }
    }

    void set_input(sensor_msgs::Image& msg) override {
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

        // Normalize the image using the mean and std for each channel
        // Mean and std are typically for [R, G, B] channels respectively
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std = {0.229f, 0.224f, 0.225f};
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            channel = (channel - mean[c]) / std[c];
        }

        cv::merge(channels, float_image);
        memcpy(_input_data, float_image.data, _input_size_float);
    }

void publish() override {
    cv::Mat depth_prediction = cv::Mat(_HEIGHT, _WIDTH, CV_32FC1, _output_data);
    if (_depth_metric_pub.getTopic() != "") {
        // Raw depth prediction (in meters, CV_32FC1)
        cv_bridge::CvImage depth_image;
        depth_image.header.stamp = ros::Time::now();
        depth_image.header.frame_id = "depth_frame";
        depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1; // 32-bit float
        depth_image.image = depth_prediction;
        _depth_metric_pub.publish(depth_image.toImageMsg());

    }

    if (_depth_vis_pub.getTopic() != "") {
        cv::Mat depth_visualized;
        double min_val, max_val;
        cv::minMaxLoc(depth_prediction, &min_val, &max_val);

        // Normalize depth image for visualization
        // Ignore values <= 0 (i.e., no depth / invalid depth)
        cv::normalize(depth_prediction, depth_visualized, 0, 255, cv::NORM_MINMAX);
        depth_visualized.convertTo(depth_visualized, CV_8UC1);

        cv_bridge::CvImage visualized_image;
        visualized_image.header.stamp = ros::Time::now();
        visualized_image.header.frame_id = "depth_frame";
        visualized_image.encoding = sensor_msgs::image_encodings::MONO8; // 8-bit grayscale for visualization
        visualized_image.image = depth_visualized;
        _depth_vis_pub.publish(visualized_image.toImageMsg());
    }
}

private:
    const size_t _HEIGHT = 518;
    const size_t _WIDTH = 518;
    cv::Size _INPUT_SIZE;
    ros::Publisher _depth_metric_pub;
    ros::Publisher _depth_vis_pub;
};

#endif // DEPTH_ANYTHING_HPP_
