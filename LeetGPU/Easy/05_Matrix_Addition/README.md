# Matrix Addition

> LeetGPU: https://leetgpu.com/challenges/matrix-addition

## 难度

Easy

## 题目描述

实现一个在 GPU 上对两个 N x N 的 32 位浮点数矩阵进行逐元素相加的程序。程序接收两个维度相同的输入矩阵，输出一个包含它们逐元素之和的矩阵。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在矩阵 `C` 中

## 示例

**示例 1：**

```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
        B = [[5.0, 6.0],
             [7.0, 8.0]]
Output: C = [[6.0, 8.0],
             [10.0, 12.0]]
```

**示例 2：**

```
Input:  A = [[1.5, 2.5, 3.5],
             [4.5, 5.5, 6.5],
             [7.5, 8.5, 9.5]]
        B = [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]]
Output: C = [[2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0],
             [8.0, 9.0, 10.0]]
```

## 约束条件

- 输入矩阵 `A` 和 `B` 维度相同
- 1 ≤ `N` ≤ 4096
- 所有元素为 32 位浮点数
- 性能测试使用 `N` = 4,096

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
