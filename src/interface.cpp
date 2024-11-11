#include "usb_cam/learning/interface.hpp"
#include <NvOnnxParser.h>

using namespace nvinfer1;

void LearningInterface::load_model() {
    if (_model_path.find(".onnx") == std::string::npos) {
        std::ifstream engine_stream(_model_path, std::ios::binary);
        engine_stream.seekg(0, std::ios::end);

        const size_t model_size = engine_stream.tellg();
        engine_stream.seekg(0, std::ios::beg);

        std::unique_ptr<char[]> engine_data(new char[model_size]);
        engine_stream.read(engine_data.get(), model_size);
        engine_stream.close();

        // Create tensorrt model
        _runtime = nvinfer1::createInferRuntime(_logger);
        _engine = _runtime->deserializeCudaEngine(engine_data.get(), model_size);
        _context = _engine->createExecutionContext();

    } else {
        // Build an engine from an onnx model
        _build(_model_path);
        _save_engine(_model_path);
    }

    // Define input dimensions
    const auto input_dims = _engine->getTensorShape(_engine->getIOTensorName(0));
    const int input_h = input_dims.d[2];
    const int input_w = input_dims.d[3];

    // Create CUDA stream
    cudaStreamCreate(&_stream);

    cudaMalloc(&_buffers[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&_buffers[1], input_h * input_w * sizeof(float));

    _output_data = new float[input_h * input_w];
}

void LearningInterface::_build(std::string onnx_path) {
    auto builder = createInferBuilder(_logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);
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

bool LearningInterface::_save_engine(const std::string& onnx_path) {
    std::string engine_path;
    size_t dot_index = onnx_path.find_last_of(".");
    if (dot_index != std::string::npos) {
        engine_path = onnx_path.substr(0, dot_index) + ".engine";

    } else {
        return false;
    }

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

void LearningInterface::predict() {
    cudaMemcpyAsync(_buffers[0], _input_data, sizeof(_input_data) * sizeof(float), cudaMemcpyHostToDevice, _stream);
    _context->executeV2(_buffers);
    cudaStreamSynchronize(_stream);
    cudaMemcpyAsync(_output_data, _buffers[1], sizeof(_input_data) * sizeof(float), cudaMemcpyDeviceToHost);
}

LearningInterface::~LearningInterface() {
    cudaFree(_stream);
    cudaFree(_buffers[0]);
    cudaFree(_buffers[1]);
}
