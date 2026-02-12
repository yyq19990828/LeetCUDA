# Interleave Arrays

> LeetGPU: https://leetgpu.com/challenges/interleave-arrays

## 难度

Easy

## 题目描述

编写一个 GPU 程序，将两个 32 位浮点数数组进行交错合并。给定两个长度为 `N` 的输入数组 `A` 和 `B`，生成一个长度为 `2N` 的输出数组，其中元素交替来自两个输入数组：`[A[0], B[0], A[1], B[1], A[2], B[2], ...]`

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input:  A = [1.0, 2.0, 3.0], B = [4.0, 5.0, 6.0]
Output: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
```

**示例 2：**

```
Input:  A = [10.0, 20.0], B = [30.0, 40.0]
Output: [10.0, 30.0, 20.0, 40.0]
```

## 约束条件

- 1 ≤ `N` ≤ 50,000,000
- 性能测试使用 `N` = 25,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
```
