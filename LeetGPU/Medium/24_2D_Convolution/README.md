# 2D Convolution

> LeetGPU: https://leetgpu.com/challenges/2d-convolution

## 难度

Medium

## 题目描述

编写一个程序，在 GPU 上执行二维卷积操作。给定一个输入矩阵和一个卷积核（滤波器），计算卷积输出。卷积应使用"valid"边界条件，即仅在卷积核与输入完全重叠的位置进行计算。输入包括：`input`：一个二维 32 位浮点数矩阵，以行优先顺序表示为一维数组。`kernel`：一个二维 32 位浮点数卷积核（滤波器），同样以行优先顺序表示为一维数组。输出应写入 `output` 矩阵（也是行优先顺序的一维数组）。输出矩阵的维度为：`output_rows = input_rows - kernel_rows + 1`，`output_cols = input_cols - kernel_cols + 1`。卷积操作定义为：output[i][j] = sum(input[i+m][j+n] * kernel[m][n])，其中 m 从 0 到 kernel_rows-1，n 从 0 到 kernel_cols-1。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input:
input (3x3):
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

kernel (2x2):
[[0, 1],
 [1, 0]]

input_rows = 3, input_cols = 3
kernel_rows = 2, kernel_cols = 2

Output (2x2):
[[6, 8],
 [12, 14]]
```

**示例 2：**

```
Input:
input (4x4):
[[1, 1, 1, 1],
 [1, 2, 3, 1],
 [1, 4, 5, 1],
 [1, 1, 1, 1]]

kernel (1x3):
[[1, 0, 1]]

input_rows = 4, input_cols = 4
kernel_rows = 1, kernel_cols = 3

Output (4x2):
[[2, 2],
 [4, 3],
 [6, 5],
 [2, 2]]
```

## 约束条件

- 1 ≤ `input_rows`, `input_cols` ≤ 3072
- 1 ≤ `kernel_rows`, `kernel_cols` ≤ 31
- `kernel_rows` ≤ `input_rows`
- `kernel_cols` ≤ `input_cols`
- 性能测试使用 `input_cols` = 3,072，`input_rows` = 3,072，`kernel_cols` = 15，`kernel_rows` = 15

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}
```
