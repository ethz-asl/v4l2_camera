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

    virtual void set_input(uint8_t* input_buffer);
    virtual void get_output(uint8_t* output_buffer);
    virtual void publish();

    void load_model();
    bool run_inference();

    virtual ~LearningInterface() {
        if (_context) _context->destroy();
        if (_engine) _engine->destroy();
        if (_runtime) _runtime->destroy();

        if (_buffers[0]) cudaFree(_buffers[0]);
        if (_buffers[1]) cudaFree(_buffers[1]);

        delete[] _input_buffer;
        delete[] _output_buffer;
    }

protected:
    std::string _model_path;
    size_t input_height;
    size_t input_width;
    size_t output_height;
    size_t output_width;

private:
    float* _input_buffer = nullptr;
    float* _output_buffer = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    nvinfer1::IRuntime* _runtime = nullptr;
    std::string _model_path;
    void* _buffers[2] = { nullptr, nullptr };
};

#endif // LEARNING_INTERFACE_HPP_
