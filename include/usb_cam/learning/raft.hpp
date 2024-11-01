#ifndef RAFT_HPP_
#define RAFT_HPP_

#include "interface.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class Raft : public LearningInterface {
public:
    Raft(std::string& model_path, size_t network_height, size_t network_width) :
    model_path(model_path), network_height(network_height), network_width(network_width) {
        network_size = cv::Size(network_width, network_height);
    }

    void set_input(const uint8_t* input_buffer, size_t height, size_t width) {
        // Resize frame to network size
        cv::Mat input_frame(height, width, CV_8UC1, (void*)input_buffer);
        cv::Mat resized_frame;
        cv::resize(input_frame, resized_frame, network_size);

        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32FC1, uint8_to_float);

        cudaMemcpy(_input_buffer, float_frame.ptr<float>(), network_width * network_height * sizeof(float), cudaMemcpyHostToDevice)
    }

private:
    const size_t network_height;
    const size_t network_width;
    cv::Size network_size;
    static constexpr float uint8_to_float = 1.0f / 255.0f;
}

#endif // RAFT_HPP_