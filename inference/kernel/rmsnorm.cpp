#include "rmsnorm.h"
#include <cmath> // For sqrt
#include <stdexcept> // For std::runtime_error
#include <numeric> // For std::accumulate

namespace happy_phone_llm {
namespace kernel {

void rmsnorm(tensor::Tensor& output, const tensor::Tensor& input, const tensor::Tensor& weight, float epsilon) {
    // 1. Input validation
    if (input.dtype() != tensor::F32 || output.dtype() != tensor::F32 || weight.dtype() != tensor::F32) {
        throw std::runtime_error("RMSNorm: All tensors must be of type F32.");
    }
    if (input.shape() != output.shape()) {
        throw std::runtime_error("RMSNorm: Input and output tensor shapes must match.");
    }
    if (input.size() != weight.size()) {
        throw std::runtime_error("RMSNorm: Input and weight tensor sizes must match.");
    }

    const float* input_data = static_cast<const float*>(input.data());
    const float* weight_data = static_cast<const float*>(weight.data());
    float* output_data = static_cast<float*>(output.data());

    // 2. Calculate sum of squares
    double sum_sq = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        sum_sq += static_cast<double>(input_data[i]) * static_cast<double>(input_data[i]);
    }

    // 3. Calculate mean square
    float mean_sq = static_cast<float>(sum_sq / input.size());

    // 4. Calculate RMS
    float rms = std::sqrt(mean_sq + epsilon);

    // 5. Apply normalization
    for (size_t i = 0; i < input.size(); ++i) {
        output_data[i] = input_data[i] * weight_data[i] / rms;
    }
}

} // namespace kernel
} // namespace happy_phone_llm

