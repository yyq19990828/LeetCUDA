# Matrix Copy

> LeetGPU: https://leetgpu.com/challenges/matrix-copy

## 难度

Easy

## 题目描述

实现一个程序，在 GPU 上将一个 N x N 的 32 位浮点数矩阵从输入数组 A 复制到输出数组 B。程序应执行逐元素的直接复制，使得对于所有有效索引，B[i][j] = A[i][j]。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在矩阵 `B` 中

## 示例

**示例 1：**

```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
```

**示例 2：**

```
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
```

## 约束条件

- 1 ≤ `N` ≤ 4096
- 所有元素均为 32 位浮点数
- 性能测试使用 `N` = 4,096

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
```
