#ifndef DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_HPP_

#include "interface.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class DepthAnythingV2 : public LearningInterface {
public:
    DepthAnythingV2(std::string model_path) {
        _model_path = model_path;
    }

    void get_output(uint8_t* output_buffer) override {
        // TODO
    }

    void publish() override {
        // TODO
    }
};

#endif // DEPTH_ANYTHING_HPP_