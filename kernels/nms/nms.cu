/**
 * @file nms.cu
 * @brief NMS (Non-Maximum Suppression) 非极大值抑制 CUDA 实现
 *
 * NMS 是目标检测中用于去除冗余检测框的后处理算法。
 * 当检测器对同一个物体产生多个重叠的候选框时，NMS 会保留置信度最高的框，
 * 抑制其他重叠程度较高的框。
 */

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// ===================== 常量与宏定义 =====================

#define WARP_SIZE 32  // CUDA warp 大小，用作 block 内线程数

// 向量化内存访问宏（本文件未使用，保留供优化扩展）
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// ===================== CUDA Kernel =====================

/**
 * @brief NMS CUDA 核函数
 *
 * 每个线程处理一个边界框，检查是否应该被抑制。
 * 框已按得分降序排列，索引小的框得分更高。
 *
 * 算法思路：
 * - 对于当前框 idx，遍历所有索引更小（得分更高）的框
 * - 如果与某个保留的框 IoU > threshold，则当前框被抑制
 * - 否则当前框被保留
 *
 * @param boxes       边界框数组，形状 [N, 4]，格式 (x1, y1, x2, y2)
 *                    - (x1, y1): 左上角坐标
 *                    - (x2, y2): 右下角坐标
 * @param scores      置信度得分数组（已排序，本 kernel 中未直接使用）
 * @param keep        输出数组，1 表示保留，0 表示被抑制
 * @param num_boxes   边界框总数
 * @param iou_threshold IoU 阈值，超过此值的框将被抑制
 */
__global__ void nms_kernel(const float *boxes, const float *scores, int *keep,
                           int num_boxes, float iou_threshold) {
  // -------------------- 计算全局线程索引 --------------------
  const int threadsPerBlock = blockDim.x;
  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int idx = blockId * threadsPerBlock + threadId;  // 当前线程处理的框索引

  // 边界检查：超出框数量的线程直接返回
  if (idx >= num_boxes)
    return;

  // -------------------- 加载当前框的坐标 --------------------
  // 边界框格式: [x1, y1, x2, y2] (左上角, 右下角)
  float x1 = boxes[idx * 4 + 0];  // 左上角 x
  float y1 = boxes[idx * 4 + 1];  // 左上角 y
  float x2 = boxes[idx * 4 + 2];  // 右下角 x
  float y2 = boxes[idx * 4 + 3];  // 右下角 y
  int suppressed = 0;  // 抑制标记（未使用，可删除）

  // -------------------- 与得分更高的框比较 --------------------
  // 遍历所有索引更小的框（得分更高，因为已按得分降序排列）
  for (int i = 0; i < idx; ++i) {
    // 跳过已被抑制的框
    if (keep[i] == 0)
      continue;

    // 加载待比较框的坐标
    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    // -------------------- 计算 IoU --------------------
    // 计算交集区域的坐标
    // 交集左上角 = 两框左上角的较大值
    // 交集右下角 = 两框右下角的较小值
    float inter_x1 = max(x1, x1_i);
    float inter_y1 = max(y1, y1_i);
    float inter_x2 = min(x2, x2_i);
    float inter_y2 = min(y2, y2_i);

    // 计算交集的宽和高（若无交集则为0）
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;  // 交集面积

    // 计算两个框的面积
    float area = (x2 - x1) * (y2 - y1);        // 当前框面积
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);  // 比较框面积

    // IoU = 交集面积 / 并集面积
    // 并集面积 = 两框面积之和 - 交集面积
    float iou = inter_area / (area + area_i - inter_area);

    // -------------------- 判断是否抑制 --------------------
    // 如果与某个保留框的 IoU 超过阈值，当前框被抑制
    if (iou > iou_threshold) {
      keep[idx] = 0;  // 标记为抑制
      return;
    }
  }

  // 遍历完所有高分框都没被抑制，则保留当前框
  keep[idx] = 1;
  return;
}

// ===================== 辅助宏定义 =====================

// 将宏参数转换为字符串
#define STRINGFY(str) #str

// PyTorch 扩展绑定宏
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// 检查张量数据类型
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// ===================== PyTorch 接口函数 =====================

/**
 * @brief NMS PyTorch 接口
 *
 * 对输入的边界框执行非极大值抑制。
 *
 * @param boxes        边界框张量，形状 [N, 4]，格式 (x1, y1, x2, y2)
 * @param scores       置信度得分张量，形状 [N]
 * @param iou_threshold IoU 阈值，默认常用 0.5
 * @return torch::Tensor 保留框的索引（相对于排序后的顺序）
 *
 * 处理流程:
 * 1. 检查输入张量类型
 * 2. 按得分降序排序
 * 3. 启动 CUDA kernel 执行 NMS
 * 4. 收集保留框的索引并返回
 */
torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores,
                  float iou_threshold) {
  // -------------------- 输入检查 --------------------
  CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);

  const int num_boxes = boxes.size(0);  // 边界框数量

  // -------------------- 分配输出张量 --------------------
  // keep 数组：1 表示保留，0 表示抑制
  auto toption =
      torch::TensorOptions().dtype(torch::kInt32).device(boxes.device());
  auto keep = torch::empty({boxes.size(0)}, toption);

  // -------------------- 配置 CUDA 网格 --------------------
  // block 大小：32 线程（一个 warp）
  // grid 大小：向上取整，确保覆盖所有框
  dim3 block(WARP_SIZE);
  dim3 grid((num_boxes + WARP_SIZE - 1) / WARP_SIZE);

  // -------------------- 按得分排序 --------------------
  // 获取降序排序后的索引
  // stable=true 保证相同得分的框保持原始顺序
  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

  // 按排序索引重排边界框
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();

  // -------------------- 启动 CUDA Kernel --------------------
  nms_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(boxes_sorted.data_ptr()),
      reinterpret_cast<float *>(scores.data_ptr()),
      reinterpret_cast<int *>(keep.data_ptr()), num_boxes, iou_threshold);

  // -------------------- 收集保留框索引 --------------------
  // 将结果拷贝回 CPU 进行后处理
  auto keep_cpu = keep.to(torch::kCPU);

  // 遍历 keep 数组，收集值为 1 的索引
  std::vector<int> keep_indices;
  auto keep_accessor = keep_cpu.accessor<int, 1>();
  for (int i = 0; i < num_boxes; ++i) {
    if (keep_accessor[i] == 1) {
      keep_indices.push_back(i);
    }
  }

  // 返回保留框的索引张量
  return torch::tensor(keep_indices,
                       torch::TensorOptions().dtype(torch::kInt32));
}

// ===================== PyBind11 模块注册 =====================

/**
 * @brief 注册 PyTorch 扩展模块
 *
 * 使用方法 (Python):
 *   import torch
 *   from torch.utils.cpp_extension import load
 *   nms_cuda = load(name='nms', sources=['nms.cu'])
 *   keep_indices = nms_cuda.nms(boxes, scores, iou_threshold)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(nms) }
