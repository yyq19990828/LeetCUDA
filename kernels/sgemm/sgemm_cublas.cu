// =============================================================================
// cuBLAS SGEMM 封装
// =============================================================================
// 本文件提供可绑定到PyTorch的cuBLAS SGEMM操作封装。
// 提供两种变体：
//   1. cublas_sgemm: 使用CUDA核心的标准FP32 SGEMM
//   2. cublas_sgemm_tf32: 使用Tensor Core的TF32 SGEMM（Ampere及更新架构）
// =============================================================================

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <torch/extension.h>
#include <torch/types.h>

#include "cublas_v2.h"

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLAS call failed");                         \
    }                                                                          \
  } while (0)

// =============================================================================
// cublas_sgemm: 标准FP32矩阵乘法
// =============================================================================
// 计算: C = alpha * A * B + beta * C，其中 alpha=1.0, beta=0.0
//
// 矩阵布局考虑：
//   - cuBLAS假设列主序存储
//   - 我们的矩阵是行主序（PyTorch默认）
//   - 要在行主序下计算 C = A * B，我们计算 C^T = B^T * A^T
//   - 这通过以下方式实现：cublasGemmEx(N, N, N, M, K, B, A, C)
//   - 第一个'N'表示B不转置（但在我们看来是B^T）
//   - 第二个'N'表示A不转置（但在我们看来是A^T）
//
// cublasGemmEx参数说明：
//   - CUBLAS_OP_N, CUBLAS_OP_N: 不显式转置（使用行主序技巧）
//   - N, M, K: 维度（注意：由于列主序，N在前）
//   - B, N: B矩阵指针和主维度（N）
//   - A, K: A矩阵指针和主维度（K）
//   - C, N: C矩阵指针和主维度（N）
// =============================================================================
void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  // 创建cuBLAS句柄
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  // 使用默认数学模式（CUDA核心上的FP32）
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  // 缩放因子: C = 1.0 * A * B + 0.0 * C
  static float alpha = 1.0;
  static float beta = 0.0;

  // 执行GEMM: 由于行主序布局，我们计算 B^T * A^T = (A*B)^T
  // 然后以列主序存储结果，看起来就是行主序的C
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
  CHECK_CUBLAS(cublasDestroy(handle));
}

// =============================================================================
// cublas_sgemm_tf32: 使用Tensor Core的TF32矩阵乘法
// =============================================================================
// 在Tensor Core上使用TF32（TensorFloat-32）精度：
//   - 在Ampere（SM80）及更新的GPU上可用
//   - 输入：FP32，但内部尾数截断为10位
//   - 累加：FP32
//   - 相比FP32 CUDA核心提供约8倍加速，精度略有损失
//
// TF32精度特性：
//   - 指数位：8位（与FP32相同）
//   - 尾数位：10位（FP32为23位）
//   - 范围：与FP32相同
//   - 精度：约3位十进制数字（FP32约7位）
//
// 最佳使用场景：
//   - 训练深度学习模型（可接受精度损失）
//   - 性能优先且可接受轻微精度损失
// =============================================================================
void cublas_sgemm_tf32(float *A, float *B, float *C, size_t M, size_t N,
                       size_t K) {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));

  // 启用TF32 Tensor Core操作
  // 这允许cuBLAS使用TF32精度的Tensor Core
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  static float alpha = 1.0;
  static float beta = 0.0;

  // CUBLAS_GEMM_DEFAULT_TENSOR_OP提示cuBLAS优先使用Tensor Core算法
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUBLAS(cublasDestroy(handle));
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// =============================================================================
// PyTorch绑定：标准FP32 cuBLAS SGEMM
// =============================================================================
// Python中使用方法:
//   import torch
//   from torch.utils.cpp_extension import load
//   lib = load(name='sgemm_lib', sources=['sgemm_cublas.cu'], ...)
//   c = torch.zeros(M, N, device='cuda')
//   lib.sgemm_cublas(a, b, c)  # c = a @ b
// =============================================================================
void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // 验证输入张量数据类型
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

  // 提取维度：a[M,K] @ b[K,N] = c[M,N]
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  // 验证张量形状
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  // 调用cuBLAS SGEMM
  cublas_sgemm(reinterpret_cast<float *>(a.data_ptr()),
               reinterpret_cast<float *>(b.data_ptr()),
               reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

// =============================================================================
// PyTorch绑定：TF32 cuBLAS SGEMM（Tensor Core加速）
// =============================================================================
// 与sgemm_cublas接口相同，但使用TF32 Tensor Core。
// 注意：需要Ampere（SM80）或更新的GPU才能使用Tensor Core加速。
// 在较旧的GPU上会回退到标准FP32计算。
// =============================================================================
void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  // 使用TF32 Tensor Core加速的SGEMM
  cublas_sgemm_tf32(reinterpret_cast<float *>(a.data_ptr()),
                    reinterpret_cast<float *>(b.data_ptr()),
                    reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}
