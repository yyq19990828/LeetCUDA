#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
/**
 * @brief 将给定变量的地址强制类型转换为 int4 指针，并获取其第一个 int4 元素。
 *
 * 该宏定义用于将任意类型的变量 value 的地址 reinterpret_cast（重新解释）为 int4* 类型，
 * 然后通过 [0] 取出第一个 int4 元素。int4 是 CUDA 中的一个结构体，包含四个 int 分量（x, y, z, w），
 * 常用于向量化操作以提升内存访问效率。
 *
 * 注意事项：
 * 1. value 变量的内存布局必须满足 int4 对齐要求（16 字节对齐），否则可能导致未定义行为。
 * 2. 该宏通常用于需要将数据块按 int4 方式批量处理的场景，如内存拷贝或向量化计算。
 * 3. 使用时需确保 value 的类型和大小适合 int4 的 reinterpret_cast 操作。
 */
//* reinterpret_cast<...>适用于指向一个完全不同、但内存布局兼容
//*（或你明确知道如何处理其布局）的类型的指针时, 可以通过一次操作
//* 处理多个数据
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// -------------------------------------- FP32
// -------------------------------------- Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Dot Product
// grid(N/256), block(256)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x; //* 线程快内的相对位置
  int idx = blockIdx.x * NUM_THREADS + tid; //* 绝对位置
  
  //* constexpr 表示编译器在编译时就能计算出值
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  //* __shared__ 表示共享内存, 对于每个线程块, 共享内存是共享的
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE; //* 目前线程块内的warp编号
  int lane = tid % WARP_SIZE; //* 目前线程块内的warp内的线程编号
  // perform warp sync reduce.
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y +
                            reg_a.z * reg_b.z + reg_a.w * reg_b.w)
                         : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// -------------------------------------- FP16
// -------------------------------------- Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    //* __hadd() 在半精度数上执行加法操作, 效率比直接使用加法高
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    // val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  half prod_f16 = (idx < N) ? __hmul(a[idx], b[idx]) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

template <const int NUM_THREADS = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2; // 2 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enough for warp operaion.
  half2 reg_a = HALF2(a[idx]);
  half2 reg_b = HALF2(b[idx]);
  half prod_f16 =
      (idx < N) ? __hadd(__hmul(reg_a.x, reg_b.x), __hmul(reg_a.y, reg_b.y))
                : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half *a, half *b, float *y,
                                               int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8], pack_b[8];                    // 8x16 bits=128 bits.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits
  const half z = __float2half(0.0f);

  half prod_f16 = z;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_DOT_PROD_KERNEL(NT, packed_type, acc_type, element_type)        \
  dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                           \
      <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),        \
                        reinterpret_cast<element_type *>(b.data_ptr()),        \
                        prod.data_ptr<float>(), N);

#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_DOT_PROD_KERNEL(32, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_DOT_PROD_KERNEL(64, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_DOT_PROD_KERNEL(128, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_DOT_PROD_KERNEL(256, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_DOT_PROD_KERNEL(512, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_DOT_PROD_KERNEL(1024, packed_type, acc_type, element_type)          \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type,   \
                               n_elements)                                     \
  torch::Tensor dot_prod_##packed_type##_##acc_type(torch::Tensor a,           \
                                                    torch::Tensor b) {         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0); \
    auto prod = torch::zeros({1}, options);                                    \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256);                                                         \
      dim3 grid(((N + 256 - 1) / 256) / (n_elements));                         \
      dot_prod_##packed_type##_##acc_type##_kernel<256>                        \
          <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),    \
                            reinterpret_cast<element_type *>(b.data_ptr()),    \
                            prod.data_ptr<float>(), N);                        \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256);                                                       \
        dim3 grid(((N + 256 - 1) / 256) / (n_elements));                       \
        dot_prod_##packed_type##_##acc_type##_kernel<256>                      \
            <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),  \
                              reinterpret_cast<element_type *>(b.data_ptr()),  \
                              prod.data_ptr<float>(), N);                      \
      }                                                                        \
    }                                                                          \
    return prod;                                                               \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PROD(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PROD(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PROD(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PROD(f16x8_pack, f32, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x8_pack_f32)
}
