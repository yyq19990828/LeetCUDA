# Leaky ReLU

> LeetGPU: https://leetgpu.com/challenges/leaky-relu

## 难度

Easy

## 题目描述

实现一个对浮点数向量执行 Leaky ReLU 激活函数的程序。Leaky ReLU 函数定义为：当 x > 0 时，f(x) = x；当 x <= 0 时，f(x) = alpha * x，其中 alpha 是一个较小的正常数（本题中为 0.01）。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在向量 `output` 中
- 使用 alpha = 0.01 作为泄漏系数

## 示例

**示例 1：**

```
Input:  x = [1.0, -2.0, 3.0, -4.0]
  Output: y = [1.0, -0.02, 3.0, -0.04]
```

**示例 2：**

```
Input:  x = [-1.5, 0.0, 2.5, -3.0]
  Output: y = [-0.015, 0.0, 2.5, -0.03]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0
- 性能测试使用 `N` = 50,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
