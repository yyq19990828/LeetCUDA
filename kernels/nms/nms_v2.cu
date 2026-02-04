/**
 * @file nms_v2.cu
 * @brief NMS (Non-Maximum Suppression) 优化版 CUDA 实现
 *
 * 基于 PyTorch Vision 官方实现的优化策略:
 * https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu
 *
 * 相比基础版 nms.cu 的主要优化:
 * 1. 位掩码 (Bitmask) 策略: 用 unsigned long long 的每个 bit 记录 IoU 比较结果,
 *    避免基础版中线程间对 keep[] 数组的读写依赖 (race condition)
 * 2. 分块计算 (Block Tiling): 将 N×N 的 IoU 比较矩阵分成 B×B 个子块,
 *    每个 CUDA block 负责一个子块, 提高并行度
 * 3. 共享内存 (Shared Memory): 列方向的 box 数据加载到 shared memory,
 *    同一 block 内所有线程复用, 减少 global memory 访问
 * 4. GPU 端掩码展开: 用第二个 kernel 在 GPU 上直接处理位掩码生成最终结果,
 *    避免 GPU→CPU→GPU 的数据搬运开销
 */

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// ===================== 常量定义 =====================

// 每个 block 的线程数, 等于 unsigned long long 的位数 (64)
// 这样一个 unsigned long long 恰好能表示一个 block 内所有线程的比较结果
constexpr int kThreadsPerBlock = sizeof(unsigned long long) * 8; // 64

// 向上整除: ceil(n / m)
__host__ __device__ inline int CeilDiv(int n, int m) {
  return (n + m - 1) / m;
}

// ===================== 设备函数 =====================

/**
 * @brief 计算两个 box 的 IoU 并判断是否超过阈值
 *
 * @param a  第一个 box 的坐标指针 [x1, y1, x2, y2]
 * @param b  第二个 box 的坐标指针 [x1, y1, x2, y2]
 * @param threshold IoU 阈值
 * @return true 如果 IoU > threshold
 */
__device__ inline bool DevIoU(const float* a, const float* b,
                              float threshold) {
  // 计算交集区域
  float left = max(a[0], b[0]);
  float top = max(a[1], b[1]);
  float right = min(a[2], b[2]);
  float bottom = min(a[3], b[3]);

  float width = max(right - left, 0.0f);
  float height = max(bottom - top, 0.0f);
  float inter_area = width * height;

  // 计算并集面积
  float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  float area_b = (b[2] - b[0]) * (b[3] - b[1]);

  return (inter_area / (area_a + area_b - inter_area)) > threshold;
}

// ===================== Kernel 1: 计算 IoU 位掩码 =====================

/**
 * @brief 分块计算所有 box 对的 IoU, 结果存为位掩码
 *
 * 将 N 个 box 分成 col_blocks 组, 每组 kThreadsPerBlock(64) 个 box.
 * 用一个 2D grid (col_blocks × col_blocks) 来覆盖 N×N 的比较矩阵.
 *
 * 对于 grid 中的 block (col_start, row_start):
 * - 行方向: box[row_start*64 .. row_start*64+63]  (当前线程处理的 box)
 * - 列方向: box[col_start*64 .. col_start*64+63]  (被比较的 box, 加载到 shared memory)
 *
 * 每个线程输出一个 unsigned long long, 其中第 i 位为 1 表示
 * 当前 box 与列方向第 i 个 box 的 IoU > threshold.
 *
 * 内存布局: dev_mask[cur_box_idx * col_blocks + col_start]
 *
 *                col_block 0    col_block 1    col_block 2
 *  row_block 0 [ mask_00         mask_01         mask_02 ]
 *  row_block 1 [                 mask_11         mask_12 ]
 *  row_block 2 [                                 mask_22 ]
 *
 * 优化: row_start > col_start 时跳过 (对称性, 只算上三角)
 *
 * @param n_boxes       box 总数
 * @param iou_threshold IoU 阈值
 * @param dev_boxes     box 坐标 [N, 4], 已按 score 降序排列
 * @param dev_mask      输出位掩码 [N, col_blocks]
 */
__global__ void NmsKernelMask(int n_boxes, float iou_threshold,
                              const float* dev_boxes,
                              unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // 只计算上三角部分 (row_start <= col_start)
  // 因为 NMS 只关心高分 box 对低分 box 的抑制
  if (row_start > col_start) return;

  // 当前块实际处理的 box 数量 (最后一个块可能不足 64 个)
  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  // ---- 将列方向的 box 加载到 shared memory ----
  // 同一个 block 内所有线程都需要与这些 box 比较,
  // 加载到 shared memory 可以避免重复的 global memory 访问
  __shared__ float block_boxes[kThreadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    int col_box_idx = kThreadsPerBlock * col_start + threadIdx.x;
    block_boxes[threadIdx.x * 4 + 0] = dev_boxes[col_box_idx * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] = dev_boxes[col_box_idx * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] = dev_boxes[col_box_idx * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] = dev_boxes[col_box_idx * 4 + 3];
  }
  __syncthreads();

  // ---- 每个线程计算一行 box 与列方向所有 box 的 IoU ----
  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float* cur_box = dev_boxes + cur_box_idx * 4;

    unsigned long long mask_val = 0;

    // 对角块上, 跳过自身及之前的 (避免自比较和重复)
    int start = (row_start == col_start) ? (threadIdx.x + 1) : 0;

    for (int i = start; i < col_size; ++i) {
      if (DevIoU(cur_box, block_boxes + i * 4, iou_threshold)) {
        mask_val |= 1ULL << i;
      }
    }

    const int col_blocks = CeilDiv(n_boxes, kThreadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = mask_val;
  }
}

// ===================== Kernel 2: 从位掩码中提取 keep 结果 =====================

/**
 * @brief 在 GPU 端展开位掩码, 生成最终的 keep 标记
 *
 * 顺序遍历所有 box (按 score 降序), 对于每个未被抑制的 box:
 * 1. 标记为 keep
 * 2. 将其位掩码合并到 removed 数组, 抑制与之重叠的后续 box
 *
 * 这避免了将掩码拷贝到 CPU 处理再拷回 GPU 的开销.
 * 参考: https://github.com/pytorch/vision/issues/8713
 *
 * @param keep     输出 bool 数组 [N], true 表示保留
 * @param dev_mask 输入位掩码 [N, col_blocks]
 * @param n_boxes  box 总数
 */
__global__ void GatherKeepFromMask(bool* keep,
                                   const unsigned long long* dev_mask,
                                   int n_boxes) {
  const int col_blocks = CeilDiv(n_boxes, kThreadsPerBlock);
  const int tid = threadIdx.x;

  // 使用动态 shared memory 存储 "已移除" 的位掩码
  extern __shared__ unsigned long long removed[];

  // 初始化: 所有 box 都未被移除
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  // 按块顺序遍历所有 box
  for (int nblock = 0; nblock < col_blocks; ++nblock) {
    auto removed_val = removed[nblock];
    __syncthreads();

    const int i_offset = nblock * kThreadsPerBlock;

    for (int inblock = 0; inblock < kThreadsPerBlock; ++inblock) {
      const int i = i_offset + inblock;
      if (i >= n_boxes) break;

      // 检查当前 box 是否已被移除
      if (!(removed_val & (1ULL << inblock))) {
        // 未被移除 → 保留该 box
        if (tid == 0) {
          keep[i] = true;
        }

        // 将该 box 的掩码合并到 removed, 抑制后续重叠的 box
        const unsigned long long* p = dev_mask + i * col_blocks;
        for (int j = tid; j < col_blocks; j += blockDim.x) {
          if (j >= nblock) {
            removed[j] |= p[j];
          }
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

// ===================== 辅助宏 =====================

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                            \
  if (((T).options().dtype() != (th_type))) {                           \
    std::cout << "Tensor Info:" << (T).options() << std::endl;          \
    throw std::runtime_error("values must be " #th_type);               \
  }

// ===================== PyTorch 接口函数 =====================

/**
 * @brief NMS 优化版 PyTorch 接口
 *
 * 处理流程:
 * 1. 按 score 降序排序
 * 2. Kernel 1 (NmsKernelMask): 分块计算 IoU, 输出位掩码
 * 3. Kernel 2 (GatherKeepFromMask): GPU 端展开掩码, 输出 keep 标记
 * 4. 用 masked_select 提取保留的 box 索引
 *
 * @param boxes        [N, 4] float32, 格式 (x1, y1, x2, y2)
 * @param scores       [N] float32
 * @param iou_threshold IoU 阈值
 * @return torch::Tensor 保留 box 的原始索引
 */
torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores,
                  float iou_threshold) {
  CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);

  const int num_boxes = boxes.size(0);
  if (num_boxes == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
  }

  // ---- Step 1: 按 score 降序排序 ----
  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /*descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();

  const int col_blocks = CeilDiv(num_boxes, kThreadsPerBlock);

  // ---- Step 2: 分配位掩码张量 ----
  // 每个 box 对应 col_blocks 个 unsigned long long
  auto mask = torch::empty(
      {num_boxes * col_blocks},
      boxes.options().dtype(torch::kInt64));

  // ---- Step 3: 启动 Kernel 1 - 计算 IoU 位掩码 ----
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(kThreadsPerBlock);

  NmsKernelMask<<<blocks, threads>>>(
      num_boxes, iou_threshold,
      boxes_sorted.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>()));

  // ---- Step 4: 启动 Kernel 2 - GPU 端展开掩码 ----
  auto keep = torch::zeros(
      {num_boxes},
      boxes.options().dtype(torch::kBool).device(boxes.device()));

  int gather_threads = std::min(col_blocks, kThreadsPerBlock);
  int shared_mem_size = col_blocks * sizeof(unsigned long long);

  GatherKeepFromMask<<<1, gather_threads, shared_mem_size>>>(
      keep.data_ptr<bool>(),
      reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>()),
      num_boxes);

  // 检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error in nms_v2: ") + cudaGetErrorString(err));
  }

  // ---- Step 5: 提取保留 box 的索引 ----
  // masked_select 在 GPU 端完成, 无需 CPU 中转
  return order_t.masked_select(keep);
}

// ===================== PyBind11 模块注册 =====================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(nms)
}
