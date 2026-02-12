# Gaussian Error Gated Linear Unit

> LeetGPU: https://leetgpu.com/challenges/gaussian-error-gated-linear-unit

## 难度

Easy

## 题目描述

实现高斯误差门控线性单元（GEGLU）激活函数的前向传播，用于一维输入向量。给定一个形状为 [N] 的输入张量，其中 N 为元素数量，使用逐元素公式计算输出。输入和输出张量的类型必须为 `float32`。GEGLU 的定义如下：将输入 x 分成两半：x1 和 x2。对后半部分计算 GELU：GELU(x2) = (1/2) * x2 * (1 + erf(x2 / sqrt(2)))。计算 GEGLU 输出：GEGLU(x1, x2) = x1 * GELU(x2)。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  [1.0, 1.0]  (N=2)
Output: [0.8413447]
```

**示例 2：**

```
Input:  [2.0, -1.0, 1.0, 0.5]  (N=4)
Output: [1.6826895, -0.3457312]
```

## 约束条件

- 1 ≤ `N` ≤ 1,000,000
- N 为偶数
- -100.0 ≤ 输入值 ≤ 100.0
- 性能测试使用 `N` = 1,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```
