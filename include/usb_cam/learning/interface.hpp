#ifndef LEARNING_INTERFACE_HPP_
#define LEARNING_INTERFACE_HPP_

#include <cassert>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>
#include <sstream>
#include <vector>

class LearningInterface {
public:
    LearningInterface() : _model_path("") {}

    virtual void set_input(const uint8_t* input_buffer, size_t height, size_t width) = 0;
    virtual void get_output(uint8_t* output_buffer) = 0;
    virtual void publish() = 0;

    void load_model();
    bool run_inference(size_t batch_size);

    virtual ~LearningInterface() {
        // if (_context) _context->destroy();
        // if (_engine) _engine->destroy();
        // if (_runtime) _runtime->destroy();

        if (_buffers[0]) cudaFree(_buffers[0]);
        if (_buffers[1]) cudaFree(_buffers[1]);

        delete[] _input_buffer;
        delete[] _output_buffer;
    }

    float* get_input_buffer() { return _input_buffer; }
    nvinfer1::ICudaEngine* get_engine() { return _engine; }
    nvinfer1::IExecutionContext* get_context() { return _context; }
    nvinfer1::IRuntime* get_runtime() { return _runtime; }

protected:
    float* _input_buffer = nullptr;
    float* _output_buffer = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    nvinfer1::IRuntime* _runtime = nullptr;
    size_t input_height;
    size_t input_width;
    size_t output_height;
    size_t output_width;
    std::string _model_path;

private:
    void* _buffers[2] = { nullptr, nullptr };
};

#endif // LEARNING_INTERFACE_HPP_
