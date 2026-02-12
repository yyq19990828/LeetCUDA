# 3D Convolution

> LeetGPU: https://leetgpu.com/challenges/3d-convolution

## 难度

Hard

## 题目描述

实现一个执行三维卷积操作的程序。给定一个三维输入体和一个三维卷积核（滤波器），计算卷积输出。卷积使用"valid"边界条件（无填充）。对于三维卷积，位置 (i,j,k) 处的输出为：

output(i,j,k) = sum over d,r,c of input(i+d, j+r, k+c) * kernel(d, r, c)

其中 d 从 0 到 K_d-1，r 从 0 到 K_r-1，c 从 0 到 K_c-1。

输入包括：`input`：32 位浮点数的三维体，以一维数组存储（行优先，然后深度方向）。`kernel`：32 位浮点数的三维卷积核，以一维数组存储（行优先，然后深度方向）。`input_depth`、`input_rows`、`input_cols`：输入的维度。`kernel_depth`、`kernel_rows`、`kernel_cols`：卷积核的维度。输出：`output`：一维数组（行优先，然后深度方向）存储结果。输出维度：`output_depth = input_depth - kernel_depth + 1`，`output_rows = input_rows - kernel_rows + 1`，`output_cols = input_cols - kernel_cols + 1`。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 中

## 示例

**示例：**

```
Example 1: Input volume \(V \in \mathbb{R}^{3 \times 3 \times 3}\): \[ \begin{aligned} V_{d=0} &= \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \\ V_{d=1} &= \begin{bmatrix} 10 & 11 & 12 \\ 13 & 14 & 15 \\ 16 & 17 & 18 \end{bmatrix} \\ V_{d=2} &= \begin{bmatrix} 19 & 20 & 21 \\ 22 & 23 & 24 \\ 25 & 26 & 27 \end{bmatrix} \end{aligned} \] Kernel \(K \in \mathbb{R}^{2 \times 3 \times 3}\): \[ \begin{aligned} K_{d=0} &= \begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix} \\ K_{d=1} &= \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \end{aligned} \] Output \(O \in \mathbb{R}^{2 \times 1 \times 1}\): \[ [82, 163] \] Example 2: Input volume \(V \in \mathbb{R}^{2 \times 2 \times 2}\): \[ \begin{aligned} V_{d=0} &= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\ V_{d=1} &= \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \end{aligned} \] Kernel \(K \in \mathbb{R}^{2 \times 2 \times 2}\): \[ \begin{aligned} K_{d=0} &= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \\ K_{d=1} &= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \end{aligned} \] Output \(O \in \mathbb{R}^{1 \times 1 \times 1}\): \[ [36] \]
```

## 约束条件

- 1 ≤ `input_depth`, `input_rows`, `input_cols` ≤ 256
- 1 ≤ `kernel_depth`, `kernel_rows`, `kernel_cols` ≤ 5
- `kernel_depth` ≤ `input_depth`
- `kernel_rows` ≤ `input_rows`
- `kernel_cols` ≤ `input_cols`
- 性能测试使用 `input_cols` = 128, `input_rows` = 128, `kernel_cols` = 5, `kernel_rows` = 5

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {}
```
