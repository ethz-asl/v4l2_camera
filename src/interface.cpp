#include "usb_cam/learning/interface.hpp"
#include <NvOnnxParser.h>
#include <ros/ros.h>

using namespace nvinfer1;

void LearningInterface::predict() {
    if (_predict_mutex.try_lock()) {
        cudaMemcpyAsync(_buffers[0], _input_data, _input_size_float , cudaMemcpyHostToDevice, _stream);

#if NV_TENSORRT_MAJOR < 10
        _context->enqueueV2(_buffers, _stream, nullptr);
#else
        _context->executeV2(_buffers);
#endif

        cudaStreamSynchronize(_stream);
        cudaMemcpyAsync(_output_data, _buffers[1], _output_size_float, cudaMemcpyDeviceToHost);
        _predict_mutex.unlock();

    }
}

void LearningInterface::_load_model() {
    std::ifstream file_check(_model_path);
    if (!file_check.good()) {
        std::cerr << "Error: " << _model_path << " does not exist." << std::endl;
        throw std::runtime_error("Model file not found");
    }
    file_check.close();

    // Define the expected .engine path based on the .onnx model path
    std::string engine_path = _model_path.substr(0, _model_path.find_last_of('.')) + ".engine";

    if (_model_path.find(".onnx") != std::string::npos) {
        // Check if the engine file already exists
        std::ifstream engine_check(engine_path, std::ios::binary);

        if (engine_check.good()) {
            engine_check.seekg(0, std::ios::end);
            const size_t model_size = engine_check.tellg();
            engine_check.seekg(0, std::ios::beg);

            std::unique_ptr<char[]> engine_data(new char[model_size]);
            engine_check.read(engine_data.get(), model_size);
            engine_check.close();

            // Create TensorRT runtime and load engine
            _runtime = nvinfer1::createInferRuntime(_logger);
            _engine = _runtime->deserializeCudaEngine(engine_data.get(), model_size);
            _context = _engine->createExecutionContext();

        } else {
            // Build an engine from the .onnx model and save it as .engine
            _build(_model_path);
            _save_engine(engine_path);
        }

    } else {
        std::cerr << "Error: Only .onnx model files are supported." << std::endl;
        throw std::runtime_error("Unsupported model format");
    }

    // Define input dimensions
#if NV_TENSORRT_MAJOR < 10
    auto input_dims = _engine->getBindingDimensions(0);
    auto output_dims = _engine->getBindingDimensions(1);
#else
    auto input_dims = _engine->getTensorShape(_engine->getIOTensorName(0));
    auto output_dims = _engine->getTensorShape(_engine->getIOTensorName(1));
#endif

    // TODO: THis does not generalize so well
    _input_c = input_dims.d[1];
    _input_h = input_dims.d[2];
    _input_w = input_dims.d[3];
    _output_c = output_dims.d[1];
    _output_h = output_dims.d[2];
    _output_w = output_dims.d[3];
    _input_size_float = _input_c * _input_h * _input_w * sizeof(float);
    _output_size_float = _output_c * _output_h * _output_w * sizeof(float);

    cudaStreamCreate(&_stream);

    cudaMalloc(&_buffers[0], _input_size_float);
    cudaMalloc(&_buffers[1], _output_size_float);

    _input_data = new float[_input_c * _input_h * _input_w];
    _output_data = new float[_output_c * _output_h * _output_w];
}

void LearningInterface::_build(std::string onnx_path) {
    auto builder = createInferBuilder(_logger);
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
    IBuilderConfig* config = builder->createBuilderConfig();

    // TODO: What about different hardware?
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, JETSON_MEM_LIMIT_B);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, _logger);
    bool parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    _runtime = createInferRuntime(_logger);
    _engine = _runtime->deserializeCudaEngine(plan->data(), plan->size());
    _context = _engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool LearningInterface::_save_engine(const std::string& engine_path) {
    if (_engine) {
        nvinfer1::IHostMemory* data = _engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }

        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

LearningInterface::~LearningInterface() {
    cudaFree(_stream);
    cudaFree(_buffers[0]);
    cudaFree(_buffers[1]);

    delete[] _input_data;
    delete[] _output_data;
}
