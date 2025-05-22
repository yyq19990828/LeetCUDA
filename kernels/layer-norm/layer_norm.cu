#include <algorithm> // 标准库头文件，提供通用算法
#include <cuda_bf16.h> // CUDA BFloat16 数据类型支持
#include <cuda_fp16.h> // CUDA Half (FP16) 数据类型支持
#include <cuda_fp8.h> // CUDA FP8 数据类型支持 (注意：此头文件可能需要较新的 CUDA 版本)
#include <cuda_runtime.h> // CUDA 运行时 API，包含设备管理、内存管理、核函数启动等
#include <float.h> // 浮点数属性宏，如 FLT_EPSILON
#include <stdio.h> // 标准输入输出库，用于调试输出等
#include <stdlib.h> // 标准库头文件，提供通用工具函数
#include <torch/extension.h> // PyTorch C++ 扩展接口，用于将 CUDA 核函数绑定到 PyTorch
#include <torch/types.h> // PyTorch 类型定义
#include <vector> // 标准库头文件，提供动态数组容器

#define WARP_SIZE 32 // 定义 Warp 大小，通常为 32 个线程
// 定义宏，将值 reinterpret_cast 为 int4 类型
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
// 定义宏，将值 reinterpret_cast 为 float4 类型，用于向量化加载/存储
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// 定义宏，将值 reinterpret_cast 为 half2 类型，用于 FP16 向量化加载/存储
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// 定义宏，将值 reinterpret_cast 为 __nv_bfloat162 类型，用于 BFloat16 向量化加载/存储
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 定义宏，将值 reinterpret_cast 为 float4 类型，用于加载/存储 128 位数据
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32 相关函数
// -------------------------------------- Warp Reduce Sum (FP32)
// 模板函数，实现 Warp 内的 FP32 求和归约
// kWarpSize: Warp 大小，默认为 WARP_SIZE (32)
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
// 使用 #pragma unroll 指示编译器展开循环，提高性能
#pragma unroll
  // 使用 __shfl_xor_sync 进行 Warp 内数据交换和求和
  // 0xffffffff: Warp mask，表示 Warp 内所有线程都参与同步
  // val: 当前线程的值
  // mask: 参与交换的线程 ID 的 XOR 掩码
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val; // 返回 Warp 内所有线程值的总和
}

// Block Reduce Sum (FP32)
// 块内求和归约的设备端辅助函数，用于 Layer/RMS Norm/Softmax 等
// 假设 Grid 为 1D，Block 为 1D，Grid 大小为 N/256，Block 大小为 256
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // 计算 Block 中的 Warp 数量，最多不超过 32 个 Warp (受限于每个 Block 最多 1024 个线程)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  // 获取当前线程在 Block 中的 Warp ID
  int warp = threadIdx.x / WARP_SIZE;
  // 获取当前线程在 Warp 中的 Lane ID
  int lane = threadIdx.x % WARP_SIZE;
  // 声明共享内存数组，用于存储每个 Warp 的归约结果
  static __shared__ float shared[NUM_WARPS];

  // 在 Warp 内进行求和归约
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  // Warp 中的第一个线程将 Warp 的归约结果存储到共享内存
  if (lane == 0)
    shared[warp] = val;
  // 同步 Block 内所有线程，确保所有 Warp 的归约结果都已写入共享内存
  __syncthreads();
  // 让 Warp 中的前 NUM_WARPS 个线程加载共享内存中的 Warp 归约结果
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  // 在这些线程之间再次进行 Warp 内归约，得到最终的 Block 归约结果
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val; // 返回 Block 内所有线程值的总和
}

// Layer Norm 核函数 (FP32)
// 对输入张量 x 的每一行进行 Layer Normalization
// x: 输入张量，形状为 NxK (N=batch_size*seq_len, K=hidden_size)
// y: 输出张量，形状为 NxK
// g: 缩放因子 (scale)
// b: 偏移因子 (bias)
// N: 输入张量的行数
// K: 输入张量的列数 (特征维度)
// 计算公式：y' = (x - mean(x)) / std(x)，y = y' * g + b
// mean(x) = sum(x) / K
// 1 / std(x) = rsqrtf( sum( (x - mean(x))^2 ) / K )
// Grid 大小为 N，Block 大小为 K (每个 Block 处理输入张量的一行)
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b,
                                      int N, int K) {
  // 获取当前线程在 Block 中的线程 ID (0 到 K-1)
  int tid = threadIdx.x;
  // 获取当前 Block 在 Grid 中的 Block ID (0 到 N-1)
  int bid = blockIdx.x;
  // 计算当前线程处理的全局索引
  int idx = bid * blockDim.x + threadIdx.x;
  // 定义一个小的 epsilon 值，用于防止除以零
  const float epsilon = 1e-5f;

  // 声明共享内存变量，用于存储当前 Block (即当前行) 的均值
  __shared__ float s_mean;
  // 声明共享内存变量，用于存储当前 Block (即当前行) 的方差的倒数 (标准差的倒数)
  __shared__ float s_variance;
  // 加载当前线程需要处理的输入值，如果索引超出范围则加载 0.0f
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  // 在 Block 内计算当前行的值的总和
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / (float)K;
  // 同步 Block 内所有线程，确保均值已计算并存储到共享内存
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // 计算当前值与均值的差的平方，用于计算方差
  float variance = (value - s_mean) * (value - s_mean);
  // 在 Block 内计算差的平方的总和 (即方差 * K)
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数) 并存储到共享内存
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // 如果当前索引在有效范围内，则计算归一化后的值并存储到输出张量 y
  if (idx < N * K)
    y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm 核函数 (FP32, Vec4 向量化)
// 对输入张量 x 的每一行进行 Layer Normalization，使用 float4 进行向量化加载/存储
// Grid 大小为 N，Block 大小为 K/4 (每个 Block 处理输入张量的一行，每个线程处理 4 个元素)
// NUM_THREADS: Block 中的线程数，默认为 256 / 4 = 64
template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b,
                                        int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K/4-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局起始索引 (每个线程处理 4 个元素)
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  // 定义一个小的 epsilon 值，用于防止除以零
  const float epsilon = 1e-5f;

  // 声明共享内存变量，用于存储当前 Block (即当前行) 的均值
  __shared__ float s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block (即当前行) 的方差的倒数 (标准差的倒数)
  __shared__ float s_variance; // shared within block
  // 使用 FLOAT4 宏加载 4 个 FP32 值到 float4 寄存器
  float4 reg_x = FLOAT4(x[idx]);
  // 计算当前线程处理的 4 个值的总和，如果索引超出范围则总和为 0.0f
  float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
  // 在 Block 内计算当前行的值的总和
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / (float)K;
  // 同步 Block 内所有线程，确保均值已计算并存储到共享内存
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // 声明 float4 寄存器，用于存储 (x - mean) 的值
  float4 reg_x_hat;
  // 计算每个元素与均值的差
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  // 计算差的平方的总和，用于计算方差
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                   reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  // 在 Block 内计算差的平方的总和 (即方差 * K)
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数) 并存储到共享内存
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // 声明 float4 寄存器，用于存储归一化后的输出值
  float4 reg_y;
  // 计算归一化后的值：(x - mean) * std_inv * g + b
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  // 如果当前索引在有效范围内，则使用 FLOAT4 宏将 4 个 FP32 值存储到输出张量 y
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

// -------------------------------------- FP16 相关函数
// -------------------------------------- Warp Reduce Sum (FP16)
// 模板函数，实现 Warp 内的 FP16 求和归约，结果为 FP16
// kWarpSize: Warp 大小，默认为 WARP_SIZE (32)
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // 使用 __shfl_xor_sync 进行 Warp 内数据交换
    // val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask)); // 可以使用 __hadd 进行 FP16 加法
    val += __shfl_xor_sync(0xffffffff, val, mask); // 或者直接使用 + 运算符 (需要编译器支持)
  }
  return val; // 返回 Warp 内所有线程值的总和 (FP16)
}

// 模板函数，实现 Warp 内的 FP16 求和归约，结果为 FP32
// kWarpSize: Warp 大小，默认为 WARP_SIZE (32)
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  // 将 FP16 值转换为 FP32 进行求和，以提高精度
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32; // 返回 Warp 内所有线程值的总和 (FP32)
}

// Block Reduce Sum (FP16, 结果为 FP16)
// 块内求和归约的设备端辅助函数，使用 FP16 进行归约
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
  // 计算 Block 中的 Warp 数量
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  // 获取当前线程在 Block 中的 Warp ID
  int warp = threadIdx.x / WARP_SIZE;
  // 获取当前线程在 Warp 中的 Lane ID
  int lane = threadIdx.x % WARP_SIZE;
  // 声明共享内存数组，用于存储每个 Warp 的归约结果 (FP16)
  static __shared__ half shared[NUM_WARPS];
  // reduce using half dtype within warps
  // 在 Warp 内进行 FP16 求和归约
  val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
  // Warp 中的第一个线程将 Warp 的归约结果存储到共享内存
  if (lane == 0)
    shared[warp] = val;
  // 同步 Block 内所有线程
  __syncthreads();
  // 让 Warp 中的前 NUM_WARPS 个线程加载共享内存中的 Warp 归约结果
  val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
  // 在这些线程之间再次进行 Warp 内 FP16 归约，得到最终的 Block 归约结果
  val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
  return val; // half
}

// Block Reduce Sum (FP16, 结果为 FP32)
// 块内求和归约的设备端辅助函数，使用 FP32 进行归约以提高精度
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
  // 计算 Block 中的 Warp 数量
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  // 获取当前线程在 Block 中的 Warp ID
  int warp = threadIdx.x / WARP_SIZE;
  // 获取当前线程在 Warp 中的 Lane ID
  int lane = threadIdx.x % WARP_SIZE;
  // 声明共享内存数组，用于存储每个 Warp 的归约结果 (FP32)
  static __shared__ float shared[NUM_WARPS];
  // reduce using float dtype within warps
  // 在 Warp 内进行 FP16 到 FP32 的求和归约
  float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
  // Warp 中的第一个线程将 Warp 的归约结果存储到共享内存
  if (lane == 0)
    shared[warp] = val_f32;
  // 同步 Block 内所有线程
  __syncthreads();
  // 让 Warp 中的前 NUM_WARPS 个线程加载共享内存中的 Warp 归约结果
  val_f32 = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  // 在这些线程之间再次进行 Warp 内 FP32 归约，得到最终的 Block 归约结果
  val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32);
  return val_f32; // float
}

// Layer Norm 核函数 (FP16 输入/输出, FP16 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization
// x: 输入张量，形状为 NxK (FP16)
// y: 输出张量，形状为 NxK (FP16)
// g: 缩放因子 (scale, FP32)
// b: 偏移因子 (bias, FP32)
// N: 输入张量的行数
// K: 输入张量的列数 (特征维度)
// 内部计算使用 FP16
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局索引
  int idx = bid * blockDim.x + threadIdx.x;
  // 定义 FP16 类型的 epsilon 值
  const half epsilon = __float2half(1e-5f);
  // 将 FP32 类型的 g 和 b 转换为 FP16
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  // 将整数 K 转换为 FP16
  const half K_ = __int2half_rn(K);

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP16)
  __shared__ half s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP16)
  __shared__ half s_variance; // shared within block
  // 加载当前线程需要处理的输入值 (FP16)，如果索引超出范围则加载 0.0f
  half value = (idx < N * K) ? x[idx] : __float2half(0.0f); // load once only
  // 在 Block 内计算当前行的值的总和 (FP16)
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / K_;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // 计算当前值与均值的差的平方 (FP16)
  half variance = (value - s_mean) * (value - s_mean);
  // 在 Block 内计算差的平方的总和 (FP16)
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP16) 并存储到共享内存
  if (tid == 0)
    s_variance = hrsqrt(variance / (K_ + epsilon)); // hrsqrt 是 FP16 的平方根倒数函数
  // 同步 Block 内所有线程
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // 如果当前索引在有效范围内，则计算归一化后的值并存储到输出张量 y (FP16)
  if (idx < N * K) {
    // 使用 __hfma 进行 FP16 的乘加运算：(value - s_mean) * s_variance * g_ + b_
    y[idx] = __hfma((value - s_mean) * s_variance, g_, b_);
    // y[idx] = ((value - s_mean) * s_variance) * g_ + b_; // 等价的普通运算
  }
}

// Layer Norm 核函数 (FP16 输入/输出, Vec2 向量化, FP16 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization，使用 half2 进行向量化加载/存储
// Grid 大小为 N，Block 大小为 K/2 (每个 Block 处理输入张量的一行，每个线程处理 2 个元素)
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K/2-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局起始索引 (每个线程处理 2 个元素)
  int idx = (bid * blockDim.x + threadIdx.x) * 2;
  // 定义 FP16 类型的 epsilon 值
  const half epsilon = __float2half(1e-5f);
  // 将 FP32 类型的 g 和 b 转换为 FP16
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  // 将整数 K 转换为 FP16
  const half K_ = __int2half_rn(K);

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP16)
  __shared__ half s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP16)
  __shared__ half s_variance; // shared within block
  // 使用 HALF2 宏加载 2 个 FP16 值到 half2 寄存器
  half2 reg_x = HALF2(x[idx]);
  // 计算当前线程处理的 2 个值的总和 (FP16)，如果索引超出范围则总和为 0.0f
  half value = (idx < N * K) ? (reg_x.x + reg_x.y) : __float2half(0.0f);
  // 在 Block 内计算当前行的值的总和 (FP16)
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / K_;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // 声明 half2 寄存器，用于存储 (x - mean) 的值 (FP16)
  half2 reg_x_hat;
  // 计算每个元素与均值的差 (FP16)
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  // 计算差的平方的总和 (FP16)，用于计算方差
  half variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y;
  // 在 Block 内计算差的平方的总和 (FP16)
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP16) 并存储到共享内存
  if (tid == 0)
    s_variance = hrsqrt(variance / (K_ + epsilon));
  // 同步 Block 内所有线程
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // 如果当前索引在有效范围内，则计算归一化后的值并存储到输出张量 y (FP16)
  if (idx < N * K) {
    // 声明 half2 寄存器，用于存储归一化后的输出值 (FP16)
    half2 reg_y;
    // 计算归一化后的值：(x - mean) * std_inv * g + b (FP16)
    reg_y.x = __hfma(reg_x_hat.x * s_variance, g_, b_);
    reg_y.y = __hfma(reg_x_hat.y * s_variance, g_, b_);
    // 使用 HALF2 宏将 2 个 FP16 值存储到输出张量 y
    HALF2(y[idx]) = reg_y;
  }
}

// 定义宏，计算 half2 向量的两个元素的和，并处理边界条件
#define HALF2_SUM(reg, i)                                                      \
  (((idx + (i)) < N * K) ? ((reg).x + (reg).y) : __float2half(0.0f))

// 定义宏，计算 half2 向量的每个元素与均值的差
#define HALF2_SUB(reg_y, reg_x)                                                \
  (reg_y).x = (reg_x).x - s_mean;                                              \
  (reg_y).y = (reg_x).y - s_mean;

// 定义宏，计算 half2 向量的每个元素与均值的差的平方的总和，并处理边界条件
#define HALF2_VARIANCE(reg, i)                                                 \
  (((idx + (i)) < N * K) ? ((reg).x * (reg).x + (reg).y * (reg).y)             \
                         : __float2half(0.0f))

// 定义宏，对 half2 向量进行 Layer Normalization 计算
#define HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)                                 \
  (reg_y).x = __hfma((reg_x).x * s_variance, g_, b_);                          \
  (reg_y).y = __hfma((reg_x).y * s_variance, g_, b_);

// Layer Norm 核函数 (FP16 输入/输出, Vec8 向量化, FP16 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization，使用 half2 宏加载 4 次实现 Vec8 效果
// Grid 大小为 N，Block 大小为 K/8 (每个 Block 处理输入张量的一行，每个线程处理 8 个元素)
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K/8-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局起始索引 (每个线程处理 8 个元素)
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  // 定义 FP16 类型的 epsilon 值
  const half epsilon = __float2half(1e-5f);
  // 将 FP32 类型的 g 和 b 转换为 FP16
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  // 将整数 K 转换为 FP16
  const half K_ = __int2half_rn(K);

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP16)
  __shared__ half s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP16)
  __shared__ half s_variance; // shared within block
  // 使用 HALF2 宏加载 4 个 half2 向量，共 8 个 FP16 值
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  // 计算这 8 个值的总和，使用 HALF2_SUM 宏并处理边界条件
  half value = HALF2_SUM(reg_x_0, 0);
  value += HALF2_SUM(reg_x_1, 2);
  value += HALF2_SUM(reg_x_2, 4);
  value += HALF2_SUM(reg_x_3, 6);

  // 在 Block 内计算当前行的值的总和 (FP16)
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / K_;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // manual unroll
  // 声明 half2 寄存器，用于存储 (x - mean) 的值 (FP16)
  half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
  // 计算每个 half2 向量与均值的差，使用 HALF2_SUB 宏
  HALF2_SUB(reg_x_hat_0, reg_x_0);
  HALF2_SUB(reg_x_hat_1, reg_x_1);
  HALF2_SUB(reg_x_hat_2, reg_x_2);
  HALF2_SUB(reg_x_hat_3, reg_x_3);

  // 计算这 8 个值的差的平方的总和，使用 HALF2_VARIANCE 宏并处理边界条件
  half variance = HALF2_VARIANCE(reg_x_hat_0, 0);
  variance += HALF2_VARIANCE(reg_x_hat_1, 2);
  variance += HALF2_VARIANCE(reg_x_hat_2, 4);
  variance += HALF2_VARIANCE(reg_x_hat_3, 6);

  // 在 Block 内计算差的平方的总和 (FP16)
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP16) 并存储到共享内存
  if (tid == 0)
    s_variance = hrsqrt(variance / (K_ + epsilon));
  // 同步 Block 内所有线程
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // manual unroll
  // 声明 half2 寄存器，用于存储归一化后的输出值 (FP16)
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  // 对每个 half2 向量进行 Layer Normalization 计算，使用 HALF2_LAYER_NORM 宏
  HALF2_LAYER_NORM(reg_y_0, reg_x_hat_0, g_, b_);
  HALF2_LAYER_NORM(reg_y_1, reg_x_hat_1, g_, b_);
  HALF2_LAYER_NORM(reg_y_2, reg_x_hat_2, g_, b_);
  HALF2_LAYER_NORM(reg_y_3, reg_x_hat_3, g_, b_);

  // 如果当前索引在有效范围内，则使用 HALF2 宏将 half2 向量存储到输出张量 y
  if ((idx + 0) < N * K) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N * K) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N * K) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N * K) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

// Layer Norm 核函数 (FP16 输入/输出, FP32 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization
// x: 输入张量，形状为 NxK (FP16)
// y: 输出张量，形状为 NxK (FP16)
// g: 缩放因子 (scale, FP32)
// b: 偏移因子 (bias, FP32)
// N: 输入张量的行数
// K: 输入张量的列数 (特征维度)
// 内部计算使用 FP32 以提高精度
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局索引
  int idx = bid * blockDim.x + threadIdx.x;
  // 定义 FP32 类型的 epsilon 值
  const float epsilon = 1e-5f;

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP32)
  __shared__ float s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP32)
  __shared__ float s_variance; // shared within block
  // 加载当前线程需要处理的输入值 (FP16) 并转换为 FP32，如果索引超出范围则加载 0.0f
  float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f; // load once only
  // 在 Block 内计算当前行的值的总和 (FP32)
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / (float)K;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  // 计算当前值与均值的差的平方 (FP32)
  float variance = (value - s_mean) * (value - s_mean);
  // 在 Block 内计算差的平方的总和 (FP32)
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP32) 并存储到共享内存
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  // 如果当前索引在有效范围内，则计算归一化后的值 (FP32) 并转换为 FP16 存储到输出张量 y
  if (idx < N * K) {
    // x*y + z -> x'*g + b
    // 使用 __fmaf_rn 进行 FP32 的乘加运算，并使用 __float2half 转换为 FP16
    y[idx] = __float2half(__fmaf_rn(((value - s_mean) * s_variance), g, b));
  }
}

// Layer Norm 核函数 (FP16 输入/输出, Vec8 向量化加载/存储, FP16 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization，使用 128 位加载/存储实现 Vec8 效果
// Grid 大小为 N，Block 大小为 K/8 (每个 Block 处理输入张量的一行，每个线程处理 8 个元素)
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K/8-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局起始索引 (每个线程处理 8 个元素)
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  // 定义 FP16 类型的 epsilon 值
  const half epsilon = __float2half(1e-5f);
  // 将 FP32 类型的 g 和 b 转换为 FP16
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  // 将整数 K 转换为 FP16
  const half K_ = __int2half_rn(K);
  // 定义 FP16 类型的零值
  const half z_ = __float2half(0.0f);

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP16)
  __shared__ half s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP16)
  __shared__ half s_variance; // shared within block
  // temporary register(memory), .local space in ptx, addressable
  // 声明局部数组，用于存储加载的 8 个 FP16 值和计算结果
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  // 使用 LDST128BITS 宏加载 128 位数据 (8 个 FP16 值) 到 pack_x 数组
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  // 初始化求和变量
  half value = z_;
#pragma unroll
  // 循环计算这 8 个值的总和，并处理边界条件
  for (int i = 0; i < 8; ++i) {
    value += ((idx + i) < N * K ? pack_x[i] : z_);
  }
  // 在 Block 内计算当前行的值的总和 (FP16)
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / K_;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();

  // 初始化方差变量
  half variance = z_;
#pragma unroll
  // 循环计算这 8 个值的差的平方的总和，并处理边界条件
  for (int i = 0; i < 8; ++i) {
    half v_hat = pack_x[i] - s_mean;
    variance += ((idx + i) < N * K ? v_hat * v_hat : z_);
  }
  // 在 Block 内计算差的平方的总和 (FP16)
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP16) 并存储到共享内存
  if (tid == 0)
    s_variance = hrsqrt(variance / (K_ + epsilon));
  // 同步 Block 内所有线程
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();

#pragma unroll
  // 循环计算归一化后的值 (FP16) 并存储到 pack_y 数组
  for (int i = 0; i < 8; ++i) {
    // TODO: use __hfma2, __hsub2, __hmul2 here // TODO: 可以考虑使用 FP16 向量指令进一步优化
    pack_y[i] = __hfma((pack_x[i] - s_mean) * s_variance, g_, b_);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  // 如果当前索引范围有效，则使用 LDST128BITS 宏将 pack_y 数组中的 128 位数据存储到输出张量 y
  if ((idx + 7) < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  // TODO: support non 8-multiple K here // TODO: 当前实现只支持 K 是 8 的倍数，需要添加对非 8 倍数的支持
}

// Layer Norm 核函数 (FP16 输入/输出, Vec8 向量化加载/存储, FP32 内部计算)
// 对输入张量 x 的每一行进行 Layer Normalization，使用 128 位加载/存储实现 Vec8 效果
// 内部计算使用 FP32 以提高精度
// Grid 大小为 N，Block 大小为 K/8 (每个 Block 处理输入张量的一行，每个线程处理 8 个元素)
// NUM_THREADS: Block 中的线程数，默认为 256
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {
  // 获取当前线程在 Block 中的线程 ID
  int tid = threadIdx.x; // 0..K/8-1
  // 获取当前 Block 在 Grid 中的 Block ID
  int bid = blockIdx.x;  // 0..N-1
  // 计算当前线程处理的全局起始索引 (每个线程处理 8 个元素)
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  // 定义 FP32 类型的 epsilon 值
  const float epsilon = 1e-5f;

  // 声明共享内存变量，用于存储当前 Block 的均值 (FP32)
  __shared__ float s_mean;     // shared within block
  // 声明共享内存变量，用于存储当前 Block 的方差的倒数 (标准差的倒数, FP32)
  __shared__ float s_variance; // shared within block
  // temporary register(memory), .local space in ptx, addressable
  // 声明局部数组，用于存储加载的 8 个 FP16 值和计算结果
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  // 使用 LDST128BITS 宏加载 128 位数据 (8 个 FP16 值) 到 pack_x 数组
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

  // 初始化求和变量 (FP32)
  float value = 0.0f;
#pragma unroll
  // 循环计算这 8 个值的总和 (转换为 FP32)，并处理边界条件
  for (int i = 0; i < 8; ++i) {
    value += ((idx + i) < N * K ? __half2float(pack_x[i]) : 0.0f);
  }
  // 在 Block 内计算当前行的值的总和 (FP32)
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  // Block 中的第一个线程计算均值并存储到共享内存
  if (tid == 0)
    s_mean = sum / (float)K;
  // 同步 Block 内所有线程
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();

  // 初始化方差变量 (FP32)
  float variance = 0.0f;
#pragma unroll
  // 循环计算这 8 个值的差 (转换为 FP32) 的平方的总和，并处理边界条件
  for (int i = 0; i < 8; ++i) {
    float v_hat = __half2float(pack_x[i]) - s_mean;
    variance += ((idx + i) < N * K ? v_hat * v_hat : 0.0f);
  }
  // 在 Block 内计算差的平方的总和 (FP32)
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  // Block 中的第一个线程计算方差的倒数 (标准差的倒数, FP32) 并存储到共享内存
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();

#pragma unroll
  // 循环计算归一化后的值 (FP32) 并转换为 FP16 存储到 pack_y 数组
  for (int i = 0; i < 8; ++i) {
    pack_y[i] = __float2half(
        __fmaf_rn(((__half2float(pack_x[i]) - s_mean) * s_variance), g, b));
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  // 如果当前索引范围有效，则使用 LDST128BITS 宏将 pack_y 数组中的 128 位数据存储到输出张量 y
  if ((idx + 7) < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  // TODO: support non 8-multiple K here // TODO: 当前实现只支持 K 是 8 的倍数，需要添加对非 8 倍数的支持
}

// --------------------- PyTorch bindings for custom kernel -----------------------
// 定义宏，将输入的宏参数转换为字符串字面量
#define STRINGFY(str) #str
// 定义宏，用于在 PyTorch 模块中绑定函数
// func: 要绑定的函数名
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func)); // 将函数 func 绑定到 PyTorch 模块 m，并使用函数名作为 Python 中的名称和文档字符串

// 定义宏，检查 PyTorch Tensor 的数据类型是否符合预期
// T: 要检查的 Tensor
// th_type: 期望的 PyTorch 数据类型 (如 torch::kFloat32)
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type); /* 如果数据类型不匹配，抛出运行时错误 */ \
  }

// 定义宏，检查两个 PyTorch Tensor 的形状是否匹配
// T1: 第一个 Tensor
// T2: 第二个 Tensor
#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim()); /* 检查维度是否相同 */                     \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!"); /* 如果任一维度大小不匹配，抛出运行时错误 */ \
    }                                                                          \
  }

// fp32 相关宏
// 定义宏，用于启动 FP32 Layer Norm 核函数
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F32_KERNEL(K)                                        \
  layer_norm_f32_kernel<(K)><<<grid, block>>>(                                 \
      reinterpret_cast<float *>(x.data_ptr()), /* 获取输入 Tensor 的数据指针并转换为 float* */ \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K)); /* 获取输出 Tensor 的数据指针并转换为 float* */

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP32 Layer Norm 核函数
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F32_KERNEL(N, K)                                   \
  dim3 block((K)); /* 设置 Block 大小为 K */                                   \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32_KERNEL(64)                                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(128)                                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(256)                                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(512)                                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32_KERNEL(1024)                                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP32 Layer Norm Vec4 核函数
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F32x4_KERNEL(K)                                      \
  layer_norm_f32x4_kernel<(K) / 4><<<grid, block>>>(                           \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP32 Layer Norm Vec4 核函数
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)                                 \
  dim3 block((K) / 4); /* 设置 Block 大小为 K/4 */                             \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32x4_KERNEL(64) break;                                  \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(128) break;                                 \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(256) break;                                 \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(512) break;                                 \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(1024) break;                                \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(2048) break;                                \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(4096) break;                                \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*4"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// fp16 相关宏
// 定义宏，用于启动 FP16 Layer Norm 核函数 (FP16 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16F16_KERNEL(K)                                     \
  layer_norm_f16_f16_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()), /* 获取输入 Tensor 的数据指针并转换为 half* */ \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K)); /* 获取输出 Tensor 的数据指针并转换为 half* */

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm 核函数 (FP16 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)                                \
  dim3 block((K)); /* 设置 Block 大小为 K */                                   \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F16_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F16_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP16 Layer Norm 核函数 (FP32 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16F32_KERNEL(K)                                     \
  layer_norm_f16_f32_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm 核函数 (FP32 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)                                \
  dim3 block((K)); /* 设置 Block 大小为 K */                                   \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F32_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F32_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP16 Layer Norm Vec2 核函数 (FP16 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16x2F16_KERNEL(K)                                   \
  layer_norm_f16x2_f16_kernel<(K) / 2>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm Vec2 核函数 (FP16 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)                              \
  dim3 block((K) / 2); /* 设置 Block 大小为 K/2 */                             \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(2048) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*2"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP16 Layer Norm Vec8 核函数 (FP16 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16x8F16_KERNEL(K)                                   \
  layer_norm_f16x8_f16_kernel<(K) / 8>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm Vec8 核函数 (FP16 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)                              \
  dim3 block((K) / 8); /* 设置 Block 大小为 K/8 */                             \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(2048) break;                             \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(4096) break;                             \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(8192) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP16 Layer Norm Vec8 Pack 核函数 (FP16 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(K)                             \
  layer_norm_f16x8_pack_f16_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm Vec8 Pack 核函数 (FP16 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)                        \
  dim3 block((K) / 8); /* 设置 Block 大小为 K/8 */                             \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// 定义宏，用于启动 FP16 Layer Norm Vec8 Pack 核函数 (FP32 内部计算)
// K: 特征维度大小
#define LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(K)                             \
  layer_norm_f16x8_pack_f32_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

// 定义宏，根据特征维度 K 的大小分发调用不同的 FP16 Layer Norm Vec8 Pack 核函数 (FP32 内部计算)
// N: Batch size * Sequence length
// K: 特征维度大小
#define DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)                        \
  dim3 block((K) / 8); /* 设置 Block 大小为 K/8 */                             \
  dim3 grid((N)); /* 设置 Grid 大小为 N */                                     \
  switch ((K)) { /* 根据 K 的值选择对应的核函数实例化 */                       \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8"); /* 如果 K 不在支持范围内，抛出错误 */ \
    break;                                                                     \
  }

// PyTorch 绑定函数：FP32 Layer Norm
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP32)
// y: 输出 Tensor (FP32)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP32
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  // 检查输出 Tensor 的数据类型是否为 FP32
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP32 Layer Norm 核函数
  DISPATCH_LAYER_NORM_F32_KERNEL(N, K)
}

// PyTorch 绑定函数：FP32 Layer Norm (Vec4)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP32)
// y: 输出 Tensor (FP32)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f32x4(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP32
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  // 检查输出 Tensor 的数据类型是否为 FP32
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP32 Layer Norm Vec4 核函数
  DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (FP16 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm 核函数 (FP16 内部计算)
  DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (Vec2, FP16 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm Vec2 核函数 (FP16 内部计算)
  DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (Vec8, FP16 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm Vec8 核函数 (FP16 内部计算)
  DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (Vec8 Pack, FP16 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm Vec8 Pack 核函数 (FP16 内部计算)
  DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (Vec8 Pack, FP32 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm Vec8 Pack 核函数 (FP32 内部计算)
  DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)
}

// PyTorch 绑定函数：FP16 Layer Norm (FP32 内部计算)
// 接收 PyTorch Tensor 作为输入，调用对应的 CUDA 核函数
// x: 输入 Tensor (FP16)
// y: 输出 Tensor (FP16)
// g: 缩放因子 (FP32)
// b: 偏移因子 (FP32)
void layer_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  // 检查输入 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // 检查输出 Tensor 的数据类型是否为 FP16
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // 检查输入和输出 Tensor 的形状是否匹配
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  // 获取 Tensor 的维度大小
  const int N = x.size(0); // Batch size * Sequence length
  const int K = x.size(1); // 特征维度
  // 分发调用 FP16 Layer Norm 核函数 (FP32 内部计算)
  DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)
}

// 使用 PYBIND11_MODULE 宏定义 PyTorch 扩展模块
// TORCH_EXTENSION_NAME: 模块名称，由 PyTorch 编译系统定义
// m: 模块对象
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 绑定 Layer Norm 函数到 PyTorch 模块
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f32)
}
