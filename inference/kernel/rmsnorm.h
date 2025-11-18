#ifndef HAPPY_PHONE_LLM_RMSNORM_H
#define HAPPY_PHONE_LLM_RMSNORM_H

#include "tensor.h" // Assuming tensor.h provides necessary tensor definitions

namespace happy_phone_llm {
namespace kernel {

// Forward declaration for RMS normalization function
void rmsnorm(tensor::Tensor& output, const tensor::Tensor& input, const tensor::Tensor& weight, float epsilon);

} // namespace kernel
} // namespace happy_phone_llm

#endif // HAPPY_PHONE_LLM_RMSNORM_H
