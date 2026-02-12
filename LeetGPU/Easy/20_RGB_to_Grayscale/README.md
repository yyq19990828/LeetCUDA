# RGB to Grayscale

> LeetGPU: https://leetgpu.com/challenges/rgb-to-grayscale

## 难度

Easy

## 题目描述

编写一个 GPU 程序，在 GPU 上将 RGB 图像转换为灰度图像。给定一个以一维 32 位浮点数数组表示的 RGB 输入图像，使用标准的 RGB 转灰度公式计算对应的灰度图像。转换公式为：`gray = 0.299 * R + 0.587 * G + 0.114 * B`。输入数组 `input` 包含 `height * width * 3` 个元素，其中每个像素的 RGB 值连续存储（R, G, B, R, G, B, ...）。输出数组 `output` 应包含 `height * width` 个灰度值。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中
- 使用精确的系数：红色 0.299，绿色 0.587，蓝色 0.114

## 示例

**示例 1：**

```
Input:  input = [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 128.0, 128.0, 128.0], width=2, height=2
Output: output = [76.245, 149.685, 29.07, 128.0]
```

**示例 2：**

```
Input:  input = [100.0, 150.0, 200.0], width=1, height=1
Output: output = [140.75]
```

## 约束条件

- 1 ≤ `width` ≤ 4096
- 1 ≤ `height` ≤ 4096
- `width * height` ≤ 4,194,304
- 所有 RGB 值在 [0.0, 255.0] 范围内
- 性能测试使用 `height` = 2,048，`width` = 2,048

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
```
