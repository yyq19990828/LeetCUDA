# Softmax

> LeetGPU: https://leetgpu.com/challenges/softmax

## 难度

Medium

## 题目描述

编写一个程序，在 GPU 上计算 32 位浮点数数组的 softmax 函数。softmax 函数定义如下：对于长度为 n 的输入数组 x，其 softmax 值记为 sigma(x)，是一个长度为 n 的数组，其中第 i 个元素为：sigma(x)_i = e^(x_i) / sum(e^(x_j))，j 从 1 到 n。你的解决方案应通过使用"最大值技巧"来处理潜在的溢出问题，即在指数运算前将输入数组的每个元素减去最大值。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input: [1.0, 2.0, 3.0], N = 3
Output: [0.090, 0.244, 0.665]（近似值）
```

**示例 2：**

```
Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
Output: [2.04e-09, 4.52e-07, 9.99e-01, 2.26e-02, 9.77e-01]（近似值）
```

## 约束条件

- 1 ≤ `N` ≤ 500,000
- 性能测试使用 `N` = 500,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
