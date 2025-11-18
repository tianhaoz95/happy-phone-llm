#ifndef HAPPY_PHONE_LLM_MATRIX_MULTIPLICATION_H
#define HAPPY_PHONE_LLM_MATRIX_MULTIPLICATION_H

#include "tensor/tensor.h"

namespace kernel {

// Basic matrix multiplication: C = A * B
// A: M x K
// B: K x N
// C: M x N
void matmul(const tensor::Tensor& A, const tensor::Tensor& B, tensor::Tensor& C);

} // namespace kernel

#endif // HAPPY_PHONE_LLM_MATRIX_MULTIPLICATION_H
