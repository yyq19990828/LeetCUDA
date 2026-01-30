# NMS (Non-Maximum Suppression)

## 0x00 说明

包含以下内容：

- [X] nms_kernel(CPU/GPU)
- [X] PyTorch bindings

nms cuda实现是最基础的版本，根据[官方源码](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu)可以进行进一步优化。

## 0x01 NMS 算法原理

### 什么是 NMS？

**非极大值抑制 (Non-Maximum Suppression)** 是目标检测中用于去除冗余检测框的后处理算法。当检测器对同一个物体产生多个重叠的候选框时，NMS 会保留置信度最高的框，抑制其他重叠程度较高的框。

### 算法流程

```
输入: boxes (N×4), scores (N,), iou_threshold
输出: keep_indices (保留的框索引)

1. 按 scores 降序排序所有框
2. 选择得分最高的框，加入 keep 列表
3. 计算该框与剩余所有框的 IoU
4. 移除 IoU > iou_threshold 的框（被抑制）
5. 重复步骤 2-4，直到所有框处理完毕
```

### 边界框坐标格式

边界框使用 `(x1, y1, x2, y2)` 格式，表示**左上角**和**右下角**坐标：

```
图像坐标系 (y轴向下)

    O ────────────────────→ x
    │
    │   (x1, y1) ─────────────┐
    │       │                 │
    │       │      Box        │
    │       │                 │
    │       └─────────────────(x2, y2)
    │
    ↓ y

约束: x2 > x1, y2 > y1
```

### IoU (Intersection over Union) 计算

IoU 用于衡量两个边界框的重叠程度：

```
        ┌─────────────┐
        │    Box A    │
        │   ┌─────────┼─────┐
        │   │ Inter-  │     │
        └───┼─section─┘     │
            │     Box B     │
            └───────────────┘

IoU = 交集面积 / 并集面积
    = Intersection / (Area_A + Area_B - Intersection)
```

计算步骤：
```cpp
// 交集区域坐标
inter_x1 = max(box_a.x1, box_b.x1)
inter_y1 = max(box_a.y1, box_b.y1)
inter_x2 = min(box_a.x2, box_b.x2)
inter_y2 = min(box_a.y2, box_b.y2)

// 交集面积 (若无交集则为0)
inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

// IoU
iou = inter_area / (area_a + area_b - inter_area)
```

### 示例

假设检测到 5 个框，`iou_threshold = 0.5`：

```
框索引:  [0,    1,    2,    3,    4   ]
得分:    [0.9,  0.8,  0.7,  0.6,  0.5 ]

步骤1: 选择框0 (得分0.9)，计算与其他框的IoU
       - IoU(0,1)=0.7 > 0.5 → 抑制框1
       - IoU(0,2)=0.3 < 0.5 → 保留框2
       - IoU(0,3)=0.8 > 0.5 → 抑制框3
       - IoU(0,4)=0.2 < 0.5 → 保留框4

步骤2: 选择框2 (剩余最高分)，计算与框4的IoU
       - IoU(2,4)=0.6 > 0.5 → 抑制框4

结果: keep = [0, 2]
```

### CUDA 实现思路

本实现采用简化的并行策略：

```cpp
// 每个线程处理一个框
__global__ void nms_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查当前框是否被之前的框抑制
    for (int i = 0; i < idx; ++i) {
        if (keep[i] && iou(box[idx], box[i]) > threshold) {
            keep[idx] = 0;  // 被抑制
            return;
        }
    }
    keep[idx] = 1;  // 保留
}
```

**注意**: 这是基础实现，存在线程间依赖问题。官方 TorchVision 版本使用位掩码和分块策略来优化并行度。

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 nms.py
```

输出:

```bash
-------------------------------------------------------------------------------------
                                        nboxes=1024
       out_nms: ['1021 ', '1022 ', '1023 '], len of keep: 950, time:0.26456594ms
    out_nms_th: ['1021 ', '1022 ', '1023 '], len of keep: 950, time:0.19218683ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=2048
       out_nms: ['2045 ', '2046 ', '2047 '], len of keep: 1838, time:0.47256470ms
    out_nms_th: ['2044 ', '2045 ', '2047 '], len of keep: 1838, time:0.39437532ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=4096
       out_nms: ['4092 ', '4093 ', '4095 '], len of keep: 3598, time:0.89909315ms
    out_nms_th: ['4093 ', '4094 ', '4095 '], len of keep: 3598, time:1.03515625ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=8192
       out_nms: ['8189 ', '8190 ', '8191 '], len of keep: 7023, time:1.49935722ms
    out_nms_th: ['8189 ', '8190 ', '8191 '], len of keep: 7023, time:3.39094877ms
-------------------------------------------------------------------------------------
```
