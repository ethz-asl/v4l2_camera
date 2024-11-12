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

        std::vector<float> input = _preprocess(resized_image);
        memcpy(_input_data, input.data(), _input_size_float);
    }

void publish() override {
    cv::Mat depth_prediction(_HEIGHT, _WIDTH, CV_32FC1, _output_data);
    if (_depth_metric_pub.getTopic() != "") {
        // Raw depth prediction (in meters, CV_32FC1)
        cv_bridge::CvImage depth_image;
        depth_image.header.stamp = ros::Time::now();
        depth_image.header.frame_id = "depth_frame";
        depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depth_image.image = depth_prediction;
        _depth_metric_pub.publish(depth_image.toImageMsg());
    }

    if (_depth_vis_pub.getTopic() != "") {
        cv::normalize(depth_prediction, depth_prediction, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::Mat colormap;
        cv::applyColorMap(depth_prediction, colormap, cv::COLORMAP_INFERNO);
        cv::resize(colormap, colormap, cv::Size(_WIDTH, _HEIGHT));

        cv_bridge::CvImage visualized_image;
        visualized_image.header.stamp = ros::Time::now();
        visualized_image.header.frame_id = "depth_frame";
        visualized_image.encoding = sensor_msgs::image_encodings::MONO8;
        visualized_image.image = colormap;
        _depth_vis_pub.publish(visualized_image.toImageMsg());
    }
}

private:
    const size_t _HEIGHT = 518;
    const size_t _WIDTH = 518;
    const float mean[3] = { 123.675f, 116.28f, 103.53f };
    const float std[3] = { 58.395f, 57.12f, 57.375f };
    cv::Size _INPUT_SIZE;
    ros::Publisher _depth_metric_pub;
    ros::Publisher _depth_vis_pub;

    std::vector<float> _preprocess(cv::Mat& image) {
        std::tuple<cv::Mat, int, int> resized = _resize_depth(image, _input_w, _input_h);
        cv::Mat resized_image = std::get<0>(resized);
        std::vector<float> input_tensor;
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < resized_image.rows; i++) {
                for (int j = 0; j < resized_image.cols; j++) {
                    input_tensor.emplace_back(((float)resized_image.at<cv::Vec3b>(i, j)[k] - mean[k]) / std[k]);
                }
            }
        }
        return input_tensor;
    }

    std::tuple<cv::Mat, int, int> _resize_depth(cv::Mat& img, int w, int h) {
        cv::Mat result;
        int nw, nh;
        int ih = img.rows;
        int iw = img.cols;
        float aspectRatio = (float)img.cols / (float)img.rows;

        if (aspectRatio >= 1) {
            nw = w;
            nh = int(h / aspectRatio);
        } else {
            nw = int(w * aspectRatio);
            nh = h;
        }
        cv::resize(img, img, cv::Size(nw, nh));
        result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
        cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(h, w, CV_8UC3, 0.0);
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

        std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);
        return res_tuple;
    }
};

#endif // DEPTH_ANYTHING_HPP_
