# Sigmoid Linear Unit

> LeetGPU: https://leetgpu.com/challenges/sigmoid-linear-unit

## 难度

Easy

## 题目描述

实现 SiLU（Sigmoid Linear Unit）激活函数的前向传播，用于一维输入向量。给定一个形状为 [N] 的输入张量，其中 N 为元素个数，使用逐元素公式计算输出。SiLU 定义如下：sigma(x) = 1 / (1 + e^(-x))，SiLU(x) = x * sigma(x)。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  input = [0.5, 1.0, -0.5]  (N=3)
Output: output = [0.3112295, 0.731059, -0.1887705]
```

**示例 2：**

```
Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]
```

## 约束条件

- 1 ≤ `N` ≤ 10,000
- -100.0 ≤ 输入值 ≤ 100.0
- 性能测试使用 `N` = 50,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
