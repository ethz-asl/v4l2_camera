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

        _runtime = nvinfer1::createInferRuntime(_logger);
        if (_runtime != nullptr) {
            _engine = _runtime->deserializeCudaEngine(model_data.data(), model_size);
            if (_engine != nullptr) {
                _context = _engine->createExecutionContext();
                if (_context != nullptr) {
                    // Allocate buffers for input and output
                    size_t input_size;
                    size_t output_size;
                    for (int io = 0; io < _engine->getNbBindings(); io++) {
                        const char* name = _engine->getBindingName(io);
                        std::cout << io << ": " << name;
                        const nvinfer1::Dims dims = _engine->getBindingDimensions(io);

                        size_t total_dims = 1;
                        for (int d = 0; d < dims.nbDims; d++) {
                            total_dims *= dims.d[d];
                        }

                        std::cout << " size: " << total_dims << std::endl;

                        // Check if it's an input or output binding
                        if (_engine->bindingIsInput(io)) {
                            input_size = total_dims * sizeof(float);
                        } else {
                            output_size = total_dims * sizeof(float);
                        }
                    }

                    // Allocate device buffers
                    cudaMalloc(&_buffers[0], input_size);
                    cudaMalloc(&_buffers[1], output_size);

                    // Allocate CPU buffers
                    _input_buffer = new float[input_size / sizeof(float)];
                    _output_buffer = new float[output_size / sizeof(float)];

                    std::cout << "TensorRT model loaded successfully from: " << _model_path << std::endl;
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
    if (!_context->executeV2(_buffers)) {
        std::cerr << "Failed to execute inference." << std::endl;
        return false;
    }
    return true;
}
