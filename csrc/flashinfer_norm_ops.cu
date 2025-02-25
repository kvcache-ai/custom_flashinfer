/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pytorch_extension_utils.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, 
    torch::Tensor& batch_size_tensor, double eps);

void fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
    torch::Tensor& batch_size_tensor, double eps);

void gemma_rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
    double eps);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Root mean square normalization
  m.def("rmsnorm", rmsnorm);
  // Fused add root mean square normalization
  m.def("fused_add_rmsnorm", fused_add_rmsnorm);
  // Gemma Root mean square normalization
  m.def("gemma_rmsnorm", gemma_rmsnorm);
  // Gemma Fused add root mean square normalization
  m.def("gemma_fused_add_rmsnorm", gemma_fused_add_rmsnorm);
}
