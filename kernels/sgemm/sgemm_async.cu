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

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                                               \
    asm volatile(                                                                                  \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                                               \
    asm volatile(                                                                                  \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// =============================================================================
// SGEMM: 分块矩阵乘法 + 线程分块 (8x4) + K维分块 (16) + 双缓冲
// =============================================================================
// 优化技术：
//   [1] Block Tile（分块）：每个block计算C矩阵的64x64分块
//   [2] Thread Tile（线程分块）：每个线程计算8x4=32个元素，提高计算密度
//   [3] K Tile（K维分块）：K维度按BK=16进行分块
//   [4] 向量化加载：使用FLOAT4进行128位内存事务
//   [5] 转置共享内存：s_a按[BK][BM]存储，保证合并访问
//   [6] 双缓冲：计算与内存加载重叠执行
//   [7] FMA指令：使用__fmaf_rn进行融合乘加运算
// =============================================================================
// 线程块配置：
//   blockDim = (BN/TN, BM/TM) = (16, 8) = 128线程
//   gridDim = ((N+BN-1)/BN, (M+BM-1)/BM)
// =============================================================================
// 共享内存使用：
//   s_a[2][BK][BM] = 2 * 16 * 64 * 4 = 8KB
//   s_b[2][BK][BN] = 2 * 16 * 64 * 4 = 8KB
//   总计：每个block 16KB -> 每个SM可容纳8个block（128KB共享内存）
// =============================================================================
template <const int BM = 64,
    const int BN = 64,
    const int BK = 16,
    const int TM = 8,
    const int TN = 4,
    const int OFFSET = 0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // 块索引和线程索引
    // block(BN/TN, BM/TM) -> (x=16, y=8), 共128个线程
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;  // 块内线性线程ID: 0~127

    // A和B矩阵的双缓冲共享内存
    // s_a: 以转置布局[BK][BM]存储，保证计算时的合并访问
    // s_b: 以行主序布局[BK][BN]存储
    // 内存: 每个2*(16*64*4)=8KB，共16KB，每个SM可容纳8个block
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    // 用于数据加载和计算的寄存器
    float r_load_a[8];         // 加载A的临时寄存器（每个线程8个float）
    float r_comp_a[TM];        // 计算时使用的A分块寄存器
    float r_comp_b[TN];        // 计算时使用的B分块寄存器
    float r_c[TM][TN] = {0.0}; // C分块累加器（8x4 = 每线程32个元素）

    // ==========================================================================
    // 加载索引计算
    // ==========================================================================
    // 128个线程协作加载 s_a[BK=16][BM=64] 和 s_b[BK=16][BN=64]
    //
    // 对于s_a: 64行 * 16列 = 1024个float，每个线程加载8个float
    //   - 128线程 / 2 = 每个k-half有64个线程
    //   - 每个线程处理A的一行，加载8个连续元素
    int load_a_smem_m = tid / 2;                // 行索引: 0~63
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // 列索引: 0或8（前半/后半）

    // 对于s_b: 16行 * 64列 = 1024个float，每个线程加载8个float
    //   - 128线程 / 8 = 每行16个线程
    //   - 每个线程从分配的位置加载8个连续元素
    int load_b_smem_k = tid / 8;                // 行索引: 0~15
    int load_b_smem_n = (tid % 8) * 8;          // 列索引: 0,8,16,...,56

    // 当前block的全局内存基地址
    int load_a_gmem_m = by * BM + load_a_smem_m;  // A的全局行
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // B的全局列

    // ==========================================================================
    // 阶段1：初始数据加载（bk=0）到缓冲区0
    // ==========================================================================
    // 进入主循环前加载第一个BK分块（流水线预热）
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // B直接加载到共享内存（行主序，无需转置）
        // A先加载到寄存器，再转置存入共享内存
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
        // 存储时转置A：A[m][k] -> s_a[k][m]
        // 这保证了计算阶段的合并访问
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    // ==========================================================================
    // 阶段2：双缓冲主循环
    // ==========================================================================
    // bk从1开始：使用buffer[sel]进行计算，同时加载到buffer[sel_next]
    // 这样可以将内存访问延迟与计算重叠
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;       // 当前计算用的缓冲区
        int smem_sel_next = bk & 1;        // 下一次加载用的缓冲区

        // 步骤1：发起下一次迭代的全局内存加载
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

        // 步骤2：在加载进行的同时使用当前缓冲区进行计算
#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // 从共享内存加载A和B分块到寄存器
            // 线程ty计算输出分块的第[ty*TM, ty*TM+8)行
            // 线程tx计算输出分块的第[tx*TN, tx*TN+4)列
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN]);

            // 外积：每个线程计算8x4个元素
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        // 步骤3：将加载的A数据存入共享内存（转置）
        // 在现代GPU上可以与上述计算重叠
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        __syncthreads();
    }

    // ==========================================================================
    // 阶段3：处理最后一个BK分块（流水线收尾）
    // ==========================================================================
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

    // ==========================================================================
    // 阶段4：将结果存储到全局内存
    // ==========================================================================
    // 每个线程使用向量化存储写入8x4 = 32个元素
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    }
}

// =============================================================================
// SGEMM: 与上面相同，但B矩阵使用CUDA异步拷贝（cp.async）
// =============================================================================
// 与非异步版本的主要区别：
//   - 使用cp.async指令将B矩阵从全局内存加载到共享内存
//   - cp.async绕过L1缓存，直接到共享内存
//   - 允许更好地重叠内存访问和计算
//   - 需要显式的commit_group()和wait_group()进行同步
// =============================================================================
// cp.async的优势：
//   - 减少寄存器压力（无需中间寄存器）
//   - 更好的内存带宽利用率
//   - 异步执行允许更多的延迟隐藏
// =============================================================================
template <const int BM = 64,
    const int BN = 64,
    const int BK = 16,
    const int TM = 8,
    const int TN = 4,
    const int OFFSET = 0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // 块索引和线程索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 双缓冲共享内存
    // 内存布局与非异步版本相同
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    // 用于加载和计算的寄存器
    float r_load_a[8];         // 加载A的临时寄存器（需要转置）
    float r_comp_a[TM];        // 用于计算的A分块
    float r_comp_b[TN];        // 用于计算的B分块
    float r_c[TM][TN] = {0.0}; // 累加器（8x4）

    // 加载索引（与非异步版本相同）
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8;
    int load_b_smem_k = tid / 8;
    int load_b_smem_n = (tid % 8) * 8;
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // ==========================================================================
    // 阶段1：初始异步加载（bk=0）到缓冲区0
    // ==========================================================================
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // 转换为cp.async PTX指令使用的共享内存地址
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);

        // 发起2条cp.async指令，每条加载16字节 = 4个float
        // CP_ASYNC_CA: 使用L1+L2缓存，支持4/8/16字节传输
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();  // 标记异步拷贝批次结束

        // A使用常规加载（需要转置，无法直接使用cp.async）
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
        // 转置A：按列存储
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);  // 等待所有待处理的异步拷贝完成
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    }
}

// =============================================================================
// SGEMM: 分块矩阵乘法 + 线程分块 (8x8) + K维分块 (16) + 双缓冲
// =============================================================================
// 更大分块版本，提高算术强度：
//   - Block Tile：128x128（相比8x4版本的64x64）
//   - Thread Tile：8x8 = 每线程64个元素（相比8x4 = 32）
//   - 每次内存访问有更多计算
// =============================================================================
// 线程块：256线程（16x16）
// 每个线程计算8x8个元素，但分布在4个象限：
//   - r_c[0:4][0:4] -> 左上象限
//   - r_c[0:4][4:8] -> 右上象限
//   - r_c[4:8][0:4] -> 左下象限
//   - r_c[4:8][4:8] -> 右下象限
// =============================================================================
// 共享内存：2 * (16*128 + 16*128) * 4 = 每个block 32KB
// =============================================================================
template <const int BM = 128,
    const int BN = 128,
    const int BK = 16,
    const int TM = 8,
    const int TN = 8,
    const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // 块/线程索引：每个block 256线程（16x16）
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 双缓冲共享内存，可选填充以减少bank冲突
    __shared__ float s_a[2][BK][BM + OFFSET];  // 转置存储
    __shared__ float s_b[2][BK][BN + OFFSET];  // 行主序存储

    // 寄存器
    float r_load_a[8];         // 加载A用（每线程8个float）
    float r_comp_a[TM];        // 计算用A分块（8个元素）
    float r_comp_b[TN];        // 计算用B分块（8个元素）
    float r_c[TM][TN] = {0.0}; // 累加器：8x8 = 每线程64个元素

    // ==========================================================================
    // 256线程加载128x16和16x128分块的索引计算
    // ==========================================================================
    // 对于s_a[BK=16][BM=128]：128*16=2048个float，256线程，每线程8个
    //   - 每2个线程处理A的一行（16个元素分成8+8）
    int load_a_smem_m = tid / 2;                // 行：0~127
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // 列：0或8

    // 对于s_b[BK=16][BN=128]：16*128=2048个float，256线程，每线程8个
    //   - 每16个线程处理B的一行（128个元素）
    int load_b_smem_k = tid / 16;               // 行：0~15
    int load_b_smem_n = (tid % 16) * 8;         // 列：0,8,16,...,120

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

template <const int BM = 128,
    const int BN = 128,
    const int BK = 16,
    const int TM = 8,
    const int TN = 8,
    const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8]; // load 8 values per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // 8x8

    // 256 threads, tx: 0~15, ty: 0~7
    int load_a_smem_m = tid / 2;                // (0,1,2,...,128)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    int load_b_smem_k = tid / 16;               // 0~15
    int load_b_smem_n = (tid % 16) * 8;         // (0,8,16,...,128)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

// =============================================================================
// SGEMM: 分块 (128x256) + 线程分块 (8x16) + K维分块 (16) + 双缓冲
// =============================================================================
// 非对称分块，更好地覆盖N维度：
//   - Block Tile：128x256（N方向更宽）
//   - Thread Tile：8x16 = 每线程128个元素
//   - 适用于高瘦矩阵或N >> M的情况
// =============================================================================
// 线程块：256线程（16x16）
// 每个线程计算8行 × 16列 = 128个元素
// =============================================================================
// 共享内存使用：
//   - s_a: 2 * 16 * 128 * 4 = 16KB
//   - s_b: 2 * 16 * 256 * 4 = 32KB
//   - 总计：48KB（在某些架构上可能需要动态共享内存）
// =============================================================================
template <const int BM = 128,
    const int BN = 256,
    const int BK = 16,
    const int TM = 8,
    const int TN = 16,
    const int OFFSET = 0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // 块/线程索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 双缓冲共享内存
    // s_a: 2 * 16 * 128 * 4 = 16KB（转置存储）
    // s_b: 2 * 16 * 256 * 4 = 32KB（行主序存储）
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    // 8x16线程分块的寄存器
    float r_load_a[8];         // 加载A用
    float r_comp_a[TM];        // 计算用8个元素
    float r_comp_b[TN];        // 计算用16个元素
    float r_c[TM][TN] = {0.0}; // 累加器：8x16 = 128个元素

    // ==========================================================================
    // 加载索引
    // ==========================================================================
    // 对于A：每线程加载8个元素（与8x8版本相同）
    int load_a_smem_m = tid / 2;                // 行：0~127
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // 列：0或8

    // 对于B：每线程加载16个元素（256列 / 每行16线程）
    int load_b_smem_k = tid / 16;        // 行：0~15
    int load_b_smem_n = (tid % 16) * 16; // 列：0,16,32,...,240

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        // gmem -> smem
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

// b: load 16 values per thread
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
        }
// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        // gmem -> smem
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

// b: load 16 values per thread
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
        }

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
            }
#pragma unroll
            for (int r = 0; r < 16; r += 4) {
                FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
            }

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
        for (int r = 0; r < 8; r += 4) {
            FLOAT4(r_comp_a[r]) = FLOAT4(s_a[1][tk][ty * TM + r]);
        }
#pragma unroll
        for (int r = 0; r < 16; r += 4) {
            FLOAT4(r_comp_b[r]) = FLOAT4(s_b[1][tk][tx * TN + r]);
        }

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = by * BM + ty * TM + m;
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

// =============================================================================
// SGEMM: 8x16线程分块 + B矩阵异步拷贝（cp.async）
// =============================================================================
// 与非异步8x16版本相同，但B矩阵加载使用cp.async：
//   - 每线程4条cp.async指令（4 * 4个float = 16个float）
//   - 通过异步流水线获得更好的带宽利用率
// =============================================================================
template <const int BM = 128,
    const int BN = 256,
    const int BK = 16,
    const int TM = 8,
    const int TN = 16,
    const int OFFSET = 0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // 块/线程索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 双缓冲共享内存（与非异步版本布局相同）
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    // 寄存器
    float r_load_a[8];         // 加载A用（需要转置）
    float r_comp_a[TM];        // 计算用8个元素
    float r_comp_b[TN];        // 计算用16个元素
    float r_c[TM][TN] = {0.0}; // 8x16累加器

    // 加载索引（与非异步版本相同）
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8;
    int load_b_smem_k = tid / 16;
    int load_b_smem_n = (tid % 16) * 16;
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        // gmem -> smem
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // b: load 16 values per thread
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 4 cp.async issue, 16 bytes = 4 float, 4x4=16
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        // gmem -> smem
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // b: load 16 values per thread
        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 4 cp.async issue, 16 bytes = 4 float, 4x4=16
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
            }
#pragma unroll
            for (int r = 0; r < 16; r += 4) {
                FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
            }

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
        for (int r = 0; r < 8; r += 4) {
            FLOAT4(r_comp_a[r]) = FLOAT4(s_a[1][tk][ty * TM + r]);
        }
#pragma unroll
        for (int r = 0; r < 16; r += 4) {
            FLOAT4(r_comp_b[r]) = FLOAT4(s_b[1][tk][tx * TN + r]);
        }

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = by * BM + ty * TM + m;
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                                       \
    if (((T).options().dtype() != (th_type))) {                                                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl;                                 \
        throw std::runtime_error("values must be " #th_type);                                      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                                        \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                                          \
        throw std::runtime_error("Tensor size mismatch!");                                         \
    }

// 8x4, k16
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

// 8x8, k16
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

// 8x16, k16
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 16;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 16;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}
