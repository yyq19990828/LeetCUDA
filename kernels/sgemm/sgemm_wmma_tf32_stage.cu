// =============================================================================
// 使用WMMA（Warp矩阵乘累加）API和TF32精度的SGEMM
// =============================================================================
// 本文件使用NVIDIA的WMMA API和TF32（TensorFloat-32）精度在Tensor Core上实现SGEMM。
// 在Ampere（SM80）及更新的GPU上可用。
//
// 主要特性：
//   - 使用wmma::fragment进行矩阵分块
//   - TF32精度：10位尾数，8位指数
//   - 多阶段流水线（2/3/4/5阶段）配合cp.async
//   - 线程块重排以优化L2缓存
//   - 支持动态共享内存以实现更大的分块尺寸
//
// WMMA分块尺寸：16x16x8（M=16, N=16, K=8）
//   - 每个warp计算一个16x16输出分块
//   - 输入分块：A[16x8], B[8x16]
//   - 多个warp协作计算更大的block分块
// =============================================================================

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

using namespace nvcuda;  // 用于wmma命名空间

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
// 支持行主序的A和B矩阵，以便与sgemm.cu和sgemm_async.cu中使用CUDA Core的内核进行比较。
// 编译时也需要相应标志。

// 向上取整除法辅助函数
HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// =============================================================================
// f32x4_tf32x4_kernel: 将FP32转换为TF32格式
// =============================================================================
// WMMA操作前需要进行TF32转换：
//   - 使用TF32的WMMA期望输入为TF32格式
//   - 此内核将FP32尾数从23位转换为10位
//   - 指数和符号保持不变
//
// 转换会截断精度，但保持与FP32相同的范围。
// 这是GEMM计算前的预处理步骤。
//
// 每个线程处理4个float（使用float4向量化）。
// =============================================================================
__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    // 加载4个FP32值
    float4 reg_x = FLOAT4(x[idx]);

    // 将每个FP32转换为TF32
    // wmma::__float_to_tf32将尾数从23位截断为10位
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);

    // 存储转换后的值
    FLOAT4(y[idx]) = reg_y;
  }
}

// =============================================================================
// 多阶段流水线WMMA SGEMM（静态共享内存版本）
// =============================================================================
// 此内核使用WMMA API在Tensor Core上执行TF32精度的SGEMM。
//
// 架构：
//   - Block Tile：128x128（BM x BN）
//   - 每个block有8个warp（256线程）
//   - 每个warp使用WMMA计算多个16x16分块
//   - Warp排列：M方向4个warp，N方向2个warp
//   - 每个warp计算2x4个WMMA分块 = 32x64个输出元素
//
// WMMA配置：
//   - WMMA_M=16, WMMA_N=16, WMMA_K=8（标准TF32分块尺寸）
//   - 每次WMMA操作：A[16x8] × B[8x16] = C[16x16]
//   - 输入类型：wmma::precision::tf32
//   - 累加器类型：float
//
// 分块层次：
//   - Block分块：BM × BN = 128 × 128
//   - Warp分块：(WMMA_M × WARP_TILE_M) × (WMMA_N × WARP_TILE_N) = 32 × 64
//   - WMMA分块：16 × 16
//   - K分块：BK = 8
//
// 流水线阶段（K_STAGE）：
//   - stage2：双缓冲，加载与计算重叠
//   - stage3：三缓冲，更好的延迟隐藏
//   - stage4/5：更多阶段，在高延迟内存上有更好的隐藏效果
//
// 线程块重排（BLOCK_SWIZZLE）：
//   - 重新排列block执行顺序以提高L2缓存命中率
//   - 对大矩阵特别有效
//   - 参考：https://zhuanlan.zhihu.com/p/555339335
// =============================================================================
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void
sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(float *A, float *B, float *C,
                                                 int M, int N, int K) {
  // 块索引，可选重排以获得更好的L2局部性
  // 当BLOCK_SWIZZLE=true时，使用z维度重新排列block
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;

  // K分块的总数
  const int NUM_K_TILES = div_ceil(K, WMMA_K);

  // Block分块维度
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16×4×2 = 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16×2×4 = 128
  constexpr int BK = WMMA_K;                             // 8

  // 多阶段共享内存缓冲区
  // A以行主序存储，用于行主序WMMA加载
  // B以行主序存储，用于行主序WMMA加载
  // 填充（A_PAD, B_PAD）可减少bank冲突
  __shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];

  // ==========================================================================
  // 线程和warp索引（静态共享内存版本）
  // ==========================================================================
  // 256线程 = 每个block 8个warp
  // Warp排列成4×2网格（M方向4个，N方向2个）
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7，block内的warp ID
  const int warp_m = warp_id / 2;      // 0,1,2,3 - M方向的warp位置
  const int warp_n = warp_id % 2;      // 0,1 - N方向的warp位置

  // ==========================================================================
  // 共享内存加载索引
  // ==========================================================================
  // 对于s_a[BM=128][BK=8]：128行 × 8列 = 1024个float
  //   - 每行8个float，每个线程加载4个（float4）
  //   - 每行2个线程，128行 → 共256线程
  int load_smem_a_m = tid / 2;                // 行索引：0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 列索引：0或4

  // 对于s_b[BK=8][BN=128]：8行 × 128列 = 1024个float
  //   - 每行128个float，每个线程加载4个（float4）
  //   - 每行32个线程，8行 → 共256线程
  int load_smem_b_k = tid / 32;       // 行索引：0~7
  int load_smem_b_n = (tid % 32) * 4; // 列索引：0,4,8,...,124

  // 当前block的全局内存地址
  int load_gmem_a_m = by * BM + load_smem_a_m; // A的全局行
  int load_gmem_b_n = bx * BN + load_smem_b_n; // B的全局列

  // ==========================================================================
  // WMMA fragment声明
  // ==========================================================================
  // 每个warp计算 WARP_TILE_M × WARP_TILE_N = 2×4 = 8个WMMA分块
  // C_frag：输出的累加器fragment，每个16×16
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

  // 将累加器fragment初始化为零
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // ==========================================================================
  // 流水线预热：在主循环前填充K_STAGE-1个阶段
  // ==========================================================================
  // 使用cp.async进行异步的全局内存到共享内存拷贝
  // CP_ASYNC_CG：cache global（仅L2），支持16字节传输
#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // 异步拷贝A分块到共享内存
    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    // 异步拷贝B分块到共享内存
    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[k][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();  // 提交这批异步拷贝
  }

  // 等待除最后一组外的所有组完成
  // s2->剩余0组, s3->1, s4->2
  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + A_PAD);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // 确保所有内存操作完成
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // 处理最后(K_STAGE-1)个k迭代
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // 加载2个分块到寄存器，smem a -> frags a，warp_m 0~3
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // 加载4个分块到寄存器，smem b -> frags b，warp_n 0~2
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

// 最后，将结果存回C矩阵
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

// =============================================================================
// 多阶段流水线WMMA SGEMM（动态共享内存版本）
// =============================================================================
// 与静态版本算法相同，但使用动态共享内存以支持更大的分块尺寸或更多流水线阶段。
//
// 何时使用动态共享内存：
//   - 当静态共享内存超过48KB默认限制时
//   - 当需要超过4个流水线阶段时
//   - 当需要更大的填充来避免bank冲突时
//
// 共享内存估算（A_PAD=4, B_PAD=4时）：
//   - stage2: ~21KB (2×128×12×4 + 2×8×132×4)
//   - stage3: ~31KB
//   - stage4: ~41KB
//   - stage5: ~51KB
//
// 要使用超过48KB，必须在启动前调用cudaFuncSetAttribute：
//   cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size)
//
// 最大动态共享内存：
//   - Ampere: 每block 164KB（需要选择启用）
//   - Ada/Hopper: 每block最多228KB
// =============================================================================
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel(
    float *A, float *B, float *C, int M, int N, int K) {
  // 块索引，可选重排
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);

  // Block分块维度
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
  constexpr int BK = WMMA_K;                             // 8

  // ==========================================================================
  // 动态共享内存布局
  // ==========================================================================
  // 内存作为单个连续块分配，然后分区：
  //   smem[0 ... s_a_size-1]: 所有阶段的A分块
  //   smem[s_a_size ... end]: 所有阶段的B分块
  //
  // 使用1D寻址，因为动态共享内存是扁平数组
  extern __shared__ float smem[];
  float *s_a = smem;                              // A缓冲区起始
  float *s_b = smem + K_STAGE * BM * (BK + A_PAD); // B缓冲区起始

  // 阶段索引的偏移量（将1D数组视为每阶段2D）
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);  // 每个A阶段的元素数
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);  // 每个B阶段的元素数

  // ==========================================================================
  // 线程和warp索引（动态共享内存版本）
  // ==========================================================================
  // 与静态版本相同的索引
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE; // 0~7，block内的warp ID
  const int warp_m = warp_id / 2;      // 0,1,2,3 - M方向的warp位置
  const int warp_n = warp_id % 2;      // 0,1 - N方向的warp位置

  // ==========================================================================
  // 共享内存加载索引
  // ==========================================================================
  // 与静态版本相同，但1D数组的地址计算方式不同
  int load_smem_a_m = tid / 2;                // 行：0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 列：0或4
  int load_smem_b_k = tid / 32;               // 行：0~7
  int load_smem_b_n = (tid % 32) * 4;         // 列：0,4,...,124

  // 全局内存地址
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

  // 只转换一次smem基地址用于cp.async
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {         // 0, 1
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(float));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(float));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    // load stage 2, k start from 2
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(float));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(float));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      float *load_smem_a_frag_ptr =
          (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
           0); // BK=WMMA_K=8
      wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      float *load_smem_b_frag_ptr =
          (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
           warp_smem_b_n); // BK=WMMA_K=8
      wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // 确保所有内存操作完成
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  // 处理最后(K_STAGE-1)个k迭代
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // 加载2个分块到寄存器，smem a -> frags a，warp_m 0~3
        int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        float *load_smem_a_frag_ptr =
            (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) +
             0); // BK=WMMA_K=8
        wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // 加载4个分块到寄存器，smem b -> frags b，warp_n 0~2
        int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        float *load_smem_b_frag_ptr =
            (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) +
             warp_smem_b_n); // BK=WMMA_K=8
        wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
      }

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

// 最后，将结果存回C矩阵
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
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

// 128x128 w/o dynamic smem
#define LAUNCH_16168_STAGE_SWIZZLE_KERNEL(stages, stride)                      \
  {                                                                            \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<                          \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, A_PAD, B_PAD, (stages), true>                             \
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),             \
                          reinterpret_cast<float *>(b.data_ptr()),             \
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);   \
  }

#define LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(stages)                           \
  {                                                                            \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<                          \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, A_PAD, B_PAD, (stages), false>                            \
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),             \
                          reinterpret_cast<float *>(b.data_ptr()),             \
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);   \
  }

// 128x128 w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...
#define LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(stages, stride)                \
  {                                                                            \
    const int smem_max_size = ((stages) * BM * (BK + A_PAD) * sizeof(float) +  \
                               (stages) * BK * (BN + B_PAD) * sizeof(float));  \
    cudaFuncSetAttribute(                                                      \
        sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<                \
            WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,     \
            WARP_TILE_N, A_PAD, B_PAD, (stages), true>,                        \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<                    \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, A_PAD, B_PAD, (stages), true>                             \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<float *>(a.data_ptr()),                           \
            reinterpret_cast<float *>(b.data_ptr()),                           \
            reinterpret_cast<float *>(c.data_ptr()), M, N, K);                 \
  }

#define LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(stages)                     \
  {                                                                            \
    const int smem_max_size = ((stages) * BM * (BK + A_PAD) * sizeof(float) +  \
                               (stages) * BK * (BN + B_PAD) * sizeof(float));  \
    cudaFuncSetAttribute(                                                      \
        sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<                \
            WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,     \
            WARP_TILE_N, A_PAD, B_PAD, (stages), false>,                       \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<                    \
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,         \
        WARP_TILE_N, A_PAD, B_PAD, (stages), false>                            \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<float *>(a.data_ptr()),                           \
            reinterpret_cast<float *>(b.data_ptr()),                           \
            reinterpret_cast<float *>(c.data_ptr()), M, N, K);                 \
  }

// 128x128 w/o dynamic smem
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c, int stages,
                                               bool swizzle,
                                               int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  const int Na = M * K;
  const int Nb = K * N;
  constexpr int T = 256;

  f32x4_tf32x4_kernel<<<((Na + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(a.data_ptr()), Na);

  f32x4_tf32x4_kernel<<<((Nb + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(b.data_ptr()), Nb);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  // s_a 2 ways bank conflicts within warp, after pad 4 -> 2 ways bank
  // conflicts. s_b 8 ways bank conflicts within warp, after pad 4 -> 4 ways
  // bank conflicts. so, the best padding policy for s_a and s_b is A_PAD=0,
  // B_PAD=0/4/8. B_PAD consume 16x~ less smem than A_PAD, 8xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 0;
  constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  constexpr int BK = WMMA_K;
  // s2: 2*128*(8)*4=8KB,  2*8*(128+0~4)*4=8.25KB,   12~13KB
  // s3: 3*128*(8)*4=12KB, 3*8*(128+0~4)*4=12.375KB, 24~25KB
  // s4: 4*128*(8)*4=16KB, 4*8*(128+0~4)*4=16.5KB,   32~33KB

  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2:
      LAUNCH_16168_STAGE_SWIZZLE_KERNEL(2, swizzle_stride);
      break;
    case 3:
      LAUNCH_16168_STAGE_SWIZZLE_KERNEL(3, swizzle_stride);
      break;
    case 4:
      LAUNCH_16168_STAGE_SWIZZLE_KERNEL(4, swizzle_stride);
      break;
    default:
      LAUNCH_16168_STAGE_SWIZZLE_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    case 3:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(3);
      break;
    case 4:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(4);
      break;
    default:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    }
  }
}

// 128x128 with dynamic smem
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(torch::Tensor a,
                                                     torch::Tensor b,
                                                     torch::Tensor c,
                                                     int stages, bool swizzle,
                                                     int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  const int Na = M * K;
  const int Nb = K * N;
  constexpr int T = 256;

  f32x4_tf32x4_kernel<<<((Na + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(a.data_ptr()), Na);

  f32x4_tf32x4_kernel<<<((Nb + T * 4 - 1) / (T * 4)), T>>>(
      reinterpret_cast<float *>(b.data_ptr()),
      reinterpret_cast<float *>(b.data_ptr()), Nb);

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;
  constexpr int WMMA_TILE_M = 4;
  constexpr int WMMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  // s_a 2 ways bank conflicts within warp, after pad 4 -> 2 ways bank
  // conflicts. s_b 8 ways bank conflicts within warp, after pad 4 -> 4 ways
  // bank conflicts. so, the best padding policy for s_a and s_b is A_PAD=0,
  // B_PAD=0/4/8. B_PAD consume 16x~ less smem than A_PAD, 8xB_PAD vs 128xA_PAD.
  constexpr int A_PAD = 0;
  constexpr int B_PAD = 0;
  constexpr int NUM_THREADS =
      (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  constexpr int BK = WMMA_K;
  // s2: 2*128*(8)*4=8KB,  2*8*(128+0~4)*4=8.25KB,   12~13KB
  // s3: 3*128*(8)*4=12KB, 3*8*(128+0~4)*4=12.375KB, 24~25KB
  // s4: 4*128*(8)*4=16KB, 4*8*(128+0~4)*4=16.5KB,   32~33KB

  if (swizzle) {
    assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2:
      LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(2, swizzle_stride);
      break;
    case 3:
      LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(3, swizzle_stride);
      break;
    case 4:
      LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(4, swizzle_stride);
      break;
    case 5:
      LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16168_STAGE_SWIZZLE_DSMEM_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(5);
    default:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    }
  }
}
