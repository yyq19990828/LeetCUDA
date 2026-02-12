# Sparse Matrix-Vector Multiplication

> LeetGPU: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication

## 难度

Medium

## 题目描述

实现一个 GPU 程序，执行稀疏矩阵-向量乘法。给定一个维度为 M x N 的稀疏矩阵 A 和一个长度为 N 的稠密向量 x，计算乘积向量 y = A x x，其长度为 M。`A` 以行优先顺序存储。`nnz` 是 `A` 中非零元素的数量。数学上，该运算定义为：y_i = sum(A_{ij} * x_j)，其中 j 从 0 到 N-1，i = 0, 1, ..., M-1。矩阵 A 的稀疏度约为 60% - 70%。

## 实现要求

- 仅使用 GPU 原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在向量 `y` 中

## 示例

**示例：**

```
Input: Matrix A (3 x 4):
  [5.0, 0.0, 0.0, 1.0]
  [0.0, 2.0, 3.0, 0.0]
  [0.0, 0.0, 0.0, 4.0]

Vector x:
  [1.0, 2.0, 3.0, 4.0]

Output: Vector y:
  [9.0, 13.0, 16.0]
```

## 约束条件

- 1 ≤ `M`, `N` ≤ 10,000
- 矩阵 A 的稀疏度约为 60%-70%（即 60%-70% 的元素为零）
- 性能测试使用 `M` = 1,000，`N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {}
```
