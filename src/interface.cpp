#include "usb_cam/learning/interface.hpp"

void LearningInterface::load_model() {
    // Open and try to read the file
    std::ifstream file(_model_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, std::ios::end);
        const size_t model_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the model data
        std::vector<char> model_data(model_size);
        file.read(model_data.data(), model_size);
        file.close();

        _runtime = nvinfer1::createInferRuntime(nvinfer1::Logger());
        if (_runtime != nullptr) {
            _engine = _runtime->deserializeCudaEngine(model_data.data(), model_size, nullptr);

            if (_engine != nullptr) {
                _context = _engine->createExecutionContext();

                if (_context != nullptr) {
                    // Allocate buffers for input and output
                    _inputSize = _engine->getBindingDimensions(0).volume() * sizeof(float);
                    _outputSize = _engine->getBindingDimensions(1).volume() * sizeof(float);

                    // Allocate device buffers
                    cudaMalloc(&_buffers[0], _inputSize);
                    cudaMalloc(&_buffers[1], _outputSize);

                    // Allocate CPU buffers
                    _inputBuffer = new float[_inputSize / sizeof(float)];
                    _outputBuffer = new float[_outputSize / sizeof(float)];

                    std::cout << "TensorRT model loaded successfully from: " << model_path << std::endl;

                } else {
                    std::cout << "Failed to create execution context." << std::endl;
                }
            } else {
                std::cout << "Failed to create TensorRT engine." << std::endl;
            }
        } else {
            std::cout << "Failed to create TensorRT runtime." << std::endl;
        }
    } else {
        std::cout << "Failed to open model file." << std::endl;
    }
}

bool LearningInterface::run_inference(size_t batch_size) {
    if (!_context->execute(batch_size, _buffers)) {
        std::cerr << "Failed to execute inference." << std::endl;
        return false;
    }
    return true;
}
