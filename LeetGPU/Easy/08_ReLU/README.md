# ReLU

> LeetGPU: https://leetgpu.com/challenges/relu

## 难度

Easy

## 题目描述

实现一个对 32 位浮点数向量执行 ReLU（Rectified Linear Unit，修正线性单元）激活函数的程序。ReLU 函数将所有负值置为零，正值保持不变：ReLU(x) = max(0, x)。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 中

## 示例

**示例 1：**

```
Input:  input = [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]
```

**示例 2：**

```
Input:  input = [-3.5, 0.0, 4.2]
Output: output = [0.0, 0.0, 4.2]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- 性能测试使用 `N` = 25,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```
