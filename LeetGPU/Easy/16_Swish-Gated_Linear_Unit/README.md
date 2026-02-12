# Swish-Gated Linear Unit

> LeetGPU: https://leetgpu.com/challenges/swish-gated-linear-unit

## 难度

Easy

## 题目描述

实现 Swish-Gated Linear Unit（SWiGLU）激活函数的前向传播，用于一维输入向量。给定一个形状为 [N] 的输入张量，其中 N 为元素个数，使用逐元素公式计算输出。输入和输出张量的类型必须为 `float32`。SWiGLU 定义如下：将输入 x 分为两半：x1 和 x2。对前半部分计算 SiLU：SiLU(x1) = x1 * sigma(x1)，其中 sigma(x) = 1 / (1 + e^(-x))。计算 SWiGLU 输出：SWiGLU(x1, x2) = SiLU(x1) * x2。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  [1.0, 2.0, 3.0, 4.0]  (N=4)
Output: [2.1931758, 7.0463767]
```

**示例 2：**

```
Input:  [0.5, 1.0]  (N=2)
Output: [0.31122968]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000
- N 为偶数
- -100.0 ≤ 输入值 ≤ 100.0
- 性能测试使用 `N` = 100,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```
