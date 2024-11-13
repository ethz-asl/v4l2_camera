#ifndef LEARNING_INTERFACE_HPP_
#define LEARNING_INTERFACE_HPP_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <string>
#include <tuple>
#include <vector>

class LearningInterface {
public:
    LearningInterface() : _model_path("") {}

    virtual void set_input(const sensor_msgs::Image& image) = 0;
    virtual void publish() = 0;

    void predict();

    nvinfer1::ICudaEngine* get_engine() { return _engine; }
    nvinfer1::IExecutionContext* get_context() { return _context; }
    nvinfer1::IRuntime* get_runtime() { return _runtime; }

    ~LearningInterface();

protected:
    cudaStream_t _stream;
    float* _input_data = nullptr;
    float* _output_data = nullptr;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    nvinfer1::INetworkDefinition* _network;
    nvinfer1::IRuntime* _runtime;
    size_t _input_c, _input_h, _input_w, _input_size_float;
    size_t _output_c, _output_h, _output_w, _output_size_float;
    std::string _model_path;

    void _load_model();

private:
    static constexpr size_t JETSON_MEM_LIMIT_B{3ULL * 1024 * 1024 * 1024};
    std::mutex _predict_mutex;
    void* _buffers[2] = { nullptr, nullptr };


    // TODO: static?
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // Only output logs with severity greater than warning
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    } _logger;

    bool _save_engine(const std::string& onnx_path);
    void _build(std::string onnx_path);
};

#endif // LEARNING_INTERFACE_HPP_
