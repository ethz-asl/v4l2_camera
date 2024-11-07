#include <gtest/gtest.h>
#include "usb_cam/learning/raft.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <fstream>

// Define a fixture for Raft tests
class RaftTest : public ::testing::Test {
protected:
    // Initialize variables for the test
    std::string model_path = "../resources/raft_small_iter10_240x320.onnx.plan";
    size_t input_height = 224;
    size_t input_width = 224;

    Raft* raft;

    void SetUp() override {
        // Instantiate the Raft model with the test parameters
        raft = new Raft(model_path, input_height, input_width);
        raft->load_model();
    }

    void TearDown() override {
        delete raft;
    }
};

TEST_F(RaftTest, TestModelLoad) {
    // Test that the model loads successfully
    ASSERT_NE(raft->get_engine(), nullptr);
    ASSERT_NE(raft->get_context(), nullptr);
    ASSERT_NE(raft->get_runtime(), nullptr);
}

TEST_F(RaftTest, TestSetInput) {
    // Create a dummy input buffer
    std::vector<uint8_t> input_data(input_height * input_width, 128);

    // Set input and check if it is copied to the device
    raft->set_input(input_data.data(), input_height, input_width);

    // Allocate host memory to copy back the data from GPU for verification
    std::vector<float> host_input(input_height * input_width);
    cudaMemcpy(host_input.data(), raft->get_input_buffer(), input_height * input_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the data (simple check to see if values are scaled correctly)
    for (size_t i = 0; i < host_input.size(); ++i) {
        ASSERT_NEAR(host_input[i], 128.0f / 255.0f, 1e-5);
    }
}

TEST_F(RaftTest, TestRunInference) {
    // Dummy batch size
    size_t batch_size = 1;

    // Run inference
    bool success = raft->run_inference(batch_size);

    // Check if inference ran successfully
    ASSERT_TRUE(success);
}
