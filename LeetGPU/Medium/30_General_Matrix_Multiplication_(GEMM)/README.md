# General Matrix Multiplication (GEMM)

> LeetGPU: https://leetgpu.com/challenges/general-matrix-multiplication-gemm

## 难度

Medium

## 题目描述

实现一个基本的通用矩阵乘法（GEMM）。给定维度为 M x K 的矩阵 A、维度为 K x N 的矩阵 B、维度为 M x N 的输入/输出矩阵 C，以及标量乘数 alpha 和 beta，计算以下运算：C = alpha * (A x B) + beta * C_initial。输入矩阵 A、B 以及 C 的初始状态包含 16 位浮点数（FP16/`half`）。所有矩阵以行优先顺序存储。标量 alpha 和 beta 为 32 位浮点数。

## 实现要求

- 仅使用原生特性（除 WMMA 外不允许使用外部库）
- `solve` 函数签名不可修改
- 乘法过程中的累加应使用 FP32 以获得更好的精度，最终结果再转换为 FP16
- 最终结果必须以 `half` 类型存储回矩阵 `C` 中

## 示例

**示例：**

```
Input:（注意：输入矩阵 A、B、C_initial 在本题中为 FP16 类型）

Matrix A (M=2, K=3):
  [1.0, 2.0, 3.0]
  [4.0, 5.0, 6.0]

Matrix B (K=3, N=2):
  [1.0, 2.0]
  [3.0, 4.0]
  [5.0, 6.0]

Matrix C_initial (M=2, N=2):
  [1.0, 1.0]
  [1.0, 1.0]

alpha = 1.0 (FP32)
beta = 0.0 (FP32)

Output (FP16): Matrix C (M=2, N=2):
  [22.0, 28.0]
  [49.0, 64.0]
```

## 约束条件

- 16 ≤ `M`, `N`, `K` ≤ 4096
- 性能测试使用 `K` = 1,024，`M` = 1,024，`N` = 1,024

## 代码模板

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {}
```
