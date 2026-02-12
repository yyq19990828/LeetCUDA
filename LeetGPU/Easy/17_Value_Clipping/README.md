# Value Clipping

> LeetGPU: https://leetgpu.com/challenges/value-clipping

## 难度

Easy

## 题目描述

编写一个 GPU 程序，对一维输入向量执行裁剪操作。给定一个形状为 [N] 的输入张量，其中 N 为元素数量，将每个元素裁剪到指定范围 [`lo`, `hi`] 内，计算输出结果。输入和输出张量的类型必须为 `float32`。裁剪的定义如下：对于输入张量中的每个元素 `x`，将其"裁剪"到允许范围 `[lo, hi]` 内。该操作确保所有值都在指定范围内，常用于机器学习中的激活值稳定化和预量化处理。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  [1.5, -2.0, 3.0, 4.5], lo = 0.0, hi = 3.5
Output: [1.5, 0.0, 3.0, 3.5]
```

**示例 2：**

```
Input:  [-1.0, 2.0, 5.0], lo = -0.5, hi = 2.5
Output: [-0.5, 2.0, 2.5]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000
- -10^6 ≤ input[i] ≤ 10^6
- `lo` ≤ `hi`
- 性能测试使用 `N` = 100,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```
