# FP16 Batched Matrix Multiplication

> LeetGPU: https://leetgpu.com/challenges/fp16-batched-matrix-multiplication

## 难度

Medium

## 题目描述

实现 FP16 批量矩阵乘法。给定一批形状为 `[B, M, K]` 的矩阵 `A` 和一批形状为 `[B, K, N]` 的矩阵 `B`，计算形状为 `[B, M, N]` 的输出批次 `C`，使得对于每个批次索引 `b`：

C_b = A_b × B_b

所有矩阵以行优先顺序存储，使用 16 位浮点数（FP16/`half`）。乘法过程中的累加应使用 FP32 以获得更高精度，最终结果再转换为 FP16。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 乘法过程中的累加应使用 FP32 以获得更高精度，最终结果再转换为 FP16
- 最终结果必须以 `half` 类型存储在 `C` 数组中

## 示例

**示例 1：**

```
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
```

## 约束条件

- 1 ≤ `B` ≤ 128
- 1 ≤ `M`, `N`, `K` ≤ 1024
- 性能测试使用 `K` = 256, `M` = 256, `N` = 256

## 代码模板

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {}
```
