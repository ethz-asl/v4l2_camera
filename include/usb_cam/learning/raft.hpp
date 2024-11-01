#ifndef RAFT_HPP_
#define RAFT_HPP_

#include "interface.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class Raft : public LearningInterface {
public:
    Raft(std::string model_path, size_t network_height, size_t network_width) : _network_height(network_height), _network_width(network_width) {
        _model_path = model_path;
        _network_size = cv::Size(_network_width, _network_height);
    }

    void set_input(const uint8_t* input_buffer, size_t height, size_t width) override {
        // Resize frame to network size
        cv::Mat input_frame(height, width, CV_8UC1, (void*)input_buffer);
        cv::Mat resized_frame;
        cv::resize(input_frame, resized_frame, _network_size);

        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32FC1, _uint8_to_float);

        cudaMemcpy(_input_buffer, float_frame.ptr<float>(), _network_width * _network_height * sizeof(float), cudaMemcpyHostToDevice);
    }

    void get_output(uint8_t* output_buffer) override {
        // TODO
    }

    void publish() override {
        // TODO
    }


private:
    const size_t _network_height;
    const size_t _network_width;
    cv::Size _network_size;
    static constexpr float _uint8_to_float = 1.0f / 255.0f;
};

#endif // RAFT_HPP_