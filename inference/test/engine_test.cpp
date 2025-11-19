#include "engine/engine.h"
#include "kernel/rmsnorm.h"
#include "kernel/attention.h"
#include "kernel/ffn.h"
#include "tensor/tensor.h"
#include "gguf/gguf.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

// Helper to create a tensor with specified values
tensor::Tensor create_test_tensor(const std::vector<uint64_t>& shape, const std::vector<float>& values) {
    tensor::Tensor t(shape, tensor::F32);
    float* data = static_cast<float*>(t.data());
    assert(t.size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    return t;
}

// Helper to check if two tensors are approximately equal
bool are_tensors_approx_equal(const tensor::Tensor& t1, const tensor::Tensor& t2, float epsilon = 1e-4) {
    if (t1.shape() != t2.shape() || t1.dtype() != t2.dtype()) {
        std::cerr << "Shape or dtype mismatch: " << t1.shape().size() << " vs " << t2.shape().size() << std::endl;
        return false;
    }
    if (t1.dtype() != tensor::F32) { // Only checking F32 for now
        std::cerr << "Unsupported dtype for comparison." << std::endl;
        return false;
    }

    const float* data1 = static_cast<const float*>(t1.data());
    const float* data2 = static_cast<const float*>(t2.data());

    for (size_t i = 0; i < t1.size(); ++i) {
        if (std::abs(data1[i] - data2[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << data1[i] << " != " << data2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// --- Kernel Tests ---

void test_rmsnorm() {
    std::cout << "Running test_rmsnorm..." << std::endl;

    // Test case 1: Simple input
    std::vector<uint64_t> shape = {1, 4};
    tensor::Tensor input = create_test_tensor(shape, {1.0f, 2.0f, 3.0f, 4.0f});
    tensor::Tensor weight = create_test_tensor(shape, {1.0f, 1.0f, 1.0f, 1.0f});
    tensor::Tensor output(shape, tensor::F32);
    float epsilon = 1e-6f;

    // Expected values:
    // mean_sq = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // rms = sqrt(7.5 + 1e-6) approx sqrt(7.5) = 2.7386
    // output = input / rms * weight
    // output[0] = 1.0 / 2.7386 = 0.3651
    // output[1] = 2.0 / 2.7386 = 0.7303
    // output[2] = 3.0 / 2.7386 = 1.0954
    // output[3] = 4.0 / 2.7386 = 1.4606
    std::vector<float> expected_values = {0.36514841f, 0.73029682f, 1.09544523f, 1.46059364f};
    tensor::Tensor expected_output = create_test_tensor(shape, expected_values);

    happy_phone_llm::kernel::rmsnorm(output, input, weight, epsilon);
    assert(are_tensors_approx_equal(output, expected_output));

    // Test case 2: Different weights
    weight = create_test_tensor(shape, {0.5f, 1.0f, 1.5f, 2.0f});
    // Recalculate expected values
    // output[0] = 1.0 / 2.7386 * 0.5 = 0.18257
    // output[1] = 2.0 / 2.7386 * 1.0 = 0.73029
    // output[2] = 3.0 / 2.7386 * 1.5 = 1.64316
    // output[3] = 4.0 / 2.7386 * 2.0 = 2.92118
    expected_values = {0.18257420f, 0.73029682f, 1.64316782f, 2.92118728f};
    expected_output = create_test_tensor(shape, expected_values);
    happy_phone_llm::kernel::rmsnorm(output, input, weight, epsilon);
    assert(are_tensors_approx_equal(output, expected_output));

    std::cout << "test_rmsnorm passed." << std::endl;
}

void test_softmax() {
    std::cout << "Running test_softmax..." << std::endl;

    // Test case 1: 1D tensor
    std::vector<uint64_t> shape_1d = {3};
    tensor::Tensor input_1d = create_test_tensor(shape_1d, {1.0f, 2.0f, 3.0f});
    happy_phone_llm::kernel::softmax(input_1d);
    // exp(1)=2.718, exp(2)=7.389, exp(3)=20.085. Sum=30.192
    // Expected: {0.0900, 0.2447, 0.6652}
    tensor::Tensor expected_output_1d = create_test_tensor(shape_1d, {0.09003057f, 0.24472847f, 0.66524096f});
    assert(are_tensors_approx_equal(input_1d, expected_output_1d));

    // Test case 2: 2D tensor
    std::vector<uint64_t> shape_2d = {2, 3};
    tensor::Tensor input_2d = create_test_tensor(shape_2d, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    happy_phone_llm::kernel::softmax(input_2d);
    // Row 1: {0.0900, 0.2447, 0.6652}
    // Row 2: exp(4)=54.598, exp(5)=148.413, exp(6)=403.428. Sum=606.439
    // Expected Row 2: {0.08999, 0.24472, 0.66527}
    tensor::Tensor expected_output_2d = create_test_tensor(shape_2d, {
        0.09003057f, 0.24472847f, 0.66524096f,
        0.09003057f, 0.24472847f, 0.66524096f
    });
    assert(are_tensors_approx_equal(input_2d, expected_output_2d));

    std::cout << "test_softmax passed." << std::endl;
}

void test_rope() {
    std::cout << "Running test_rope..." << std::endl;

    std::vector<uint64_t> shape = {1, 4};
    tensor::Tensor input = create_test_tensor(shape, {1.0f, 2.0f, 3.0f, 4.0f});
    uint32_t position = 1;
    uint32_t n_rot = 4; // Apply to all 4 dimensions

    // Expected values for position=1, n_rot=4
    // freq_cis for i=0: 1 / 10000^(0/4) = 1.0. cos(1.0)=0.5403, sin(1.0)=0.8415
    // v0=1, v1=2
    // new_v0 = 1*0.5403 - 2*0.8415 = 0.5403 - 1.683 = -1.1427
    // new_v1 = 1*0.8415 + 2*0.5403 = 0.8415 + 1.0806 = 1.9221
    // freq_cis for i=2: 1 / 10000^(2/4) = 1 / 100 = 0.01. cos(0.01)=0.99995, sin(0.01)=0.009999
    // v0=3, v1=4
    // new_v0 = 3*0.99995 - 4*0.009999 = 2.99985 - 0.039996 = 2.959854
    // new_v1 = 3*0.009999 + 4*0.99995 = 0.029997 + 3.9998 = 4.029797

    std::vector<float> expected_values = {-1.142700f, 1.922100f, 2.959854f, 4.029797f};
    tensor::Tensor expected_output = create_test_tensor(shape, expected_values);

    happy_phone_llm::kernel::rope(input, position, n_rot);
    assert(are_tensors_approx_equal(input, expected_output, 1e-4));

    std::cout << "test_rope passed." << std::endl;
}

// --- FFN Tests ---

void test_silu() {
    std::cout << "Running test_silu..." << std::endl;

    std::vector<uint64_t> shape = {1, 3};
    tensor::Tensor input = create_test_tensor(shape, {-1.0f, 0.0f, 1.0f}); // sigmoid(-1)=0.2689, sigmoid(0)=0.5, sigmoid(1)=0.7311
    // Expected: -1 * 0.2689 = -0.2689, 0 * 0.5 = 0, 1 * 0.7311 = 0.7311
    std::vector<float> expected_values = {-0.26894142f, 0.0f, 0.73105858f};
    tensor::Tensor expected_output = create_test_tensor(shape, expected_values);

    happy_phone_llm::kernel::silu(input);
    assert(are_tensors_approx_equal(input, expected_output));

    std::cout << "test_silu passed." << std::endl;
}

void test_ffn() {
    std::cout << "Running test_ffn..." << std::endl;

    // Assuming input: [1, 2], w1: [2, 4], w3: [2, 4], w2: [4, 2]
    uint64_t n_embd = 2;
    uint64_t intermediate_size = 4;
    std::vector<uint64_t> input_shape = {1, n_embd};
    std::vector<uint64_t> w_in_shape = {n_embd, intermediate_size};
    std::vector<uint64_t> w_out_shape = {intermediate_size, n_embd};

    tensor::Tensor input = create_test_tensor(input_shape, {0.5f, 1.0f});
    tensor::Tensor w1 = create_test_tensor(w_in_shape, {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f
    });
    tensor::Tensor w3 = create_test_tensor(w_in_shape, {
        0.8f, 0.7f, 0.6f, 0.5f,
        0.4f, 0.3f, 0.2f, 0.1f
    });
    tensor::Tensor w2 = create_test_tensor(w_out_shape, {
        0.9f, 1.0f,
        1.1f, 1.2f,
        1.3f, 1.4f,
        1.5f, 1.6f
    });
    tensor::Tensor output(input_shape, tensor::F32);

    // Expected manual calculation (simplified, full calculation is complex)
    // 1. gate_proj = input @ w3
    // [0.5, 1.0] @ [[0.8, 0.7, 0.6, 0.5], [0.4, 0.3, 0.2, 0.1]]
    // gate_proj = [0.5*0.8 + 1.0*0.4,  0.5*0.7 + 1.0*0.3,  0.5*0.6 + 1.0*0.2,  0.5*0.5 + 1.0*0.1]
    //           = [0.4 + 0.4,        0.35 + 0.3,       0.3 + 0.2,        0.25 + 0.1]
    //           = [0.8, 0.65, 0.5, 0.35]
    // 2. activated_gate = SiLU(gate_proj)
    // silu(0.8) = 0.8 * sigmoid(0.8) = 0.8 * 0.6899 = 0.5519
    // silu(0.65) = 0.65 * sigmoid(0.65) = 0.65 * 0.6570 = 0.4270
    // silu(0.5) = 0.5 * sigmoid(0.5) = 0.5 * 0.6224 = 0.3112
    // silu(0.35) = 0.35 * sigmoid(0.35) = 0.35 * 0.5866 = 0.2053
    // activated_gate = [0.5519, 0.4270, 0.3112, 0.2053]
    // 3. hidden_proj = input @ w1
    // [0.5, 1.0] @ [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    // hidden_proj = [0.5*0.1 + 1.0*0.5,  0.5*0.2 + 1.0*0.6,  0.5*0.3 + 1.0*0.7,  0.5*0.4 + 1.0*0.8]
    //             = [0.05 + 0.5,       0.1 + 0.6,        0.15 + 0.7,       0.2 + 0.8]
    //             = [0.55, 0.7, 0.85, 1.0]
    // 4. intermediate = hidden_proj * activated_gate (element-wise)
    // intermediate = [0.55*0.5519, 0.7*0.4270, 0.85*0.3112, 1.0*0.2053]
    //              = [0.303545, 0.2989, 0.26452, 0.2053]
    // 5. output = intermediate @ w2
    // [0.303545, 0.2989, 0.26452, 0.2053] @ [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
    // output[0] = 0.303545*0.9 + 0.2989*1.1 + 0.26452*1.3 + 0.2053*1.5 = 0.27319 + 0.32879 + 0.343876 + 0.30795 = 1.253806
    // output[1] = 0.303545*1.0 + 0.2989*1.2 + 0.26452*1.4 + 0.2053*1.6 = 0.303545 + 0.35868 + 0.370328 + 0.32848 = 1.361033

    std::vector<float> expected_output_values = {1.2539467f, 1.3611857f};
    tensor::Tensor expected_output = create_test_tensor(input_shape, expected_output_values);

    happy_phone_llm::kernel::ffn(output, input, w1, w2, w3);
    assert(are_tensors_approx_equal(output, expected_output, 2e-4));

    std::cout << "test_ffn passed." << std::endl;
}

// --- Main Test Runner ---

int main() {
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_silu();
    test_ffn();
    std::cout << "All kernel tests passed!" << std::endl;
    return 0;
}
