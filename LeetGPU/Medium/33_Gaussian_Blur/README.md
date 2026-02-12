# Gaussian Blur

> LeetGPU: https://leetgpu.com/challenges/gaussian-blur

## 难度

Medium

## 题目描述

实现一个程序，将高斯模糊滤波器应用于二维图像。给定一个以浮点数组表示的输入图像和一个高斯核，程序应计算图像与核的卷积。所有输入和输出以行优先顺序存储。高斯模糊通过将每个像素与其邻域像素的加权平均进行卷积来实现，其中权重由高斯核决定。对于位置 (i, j) 处的每个输出像素，其值计算为：output[i, j] = sum(sum(input[i+m, j+n] * kernel[m+k_h/2, n+k_w/2]))，其中 m 的范围为 -k_h/2 到 k_h/2，n 的范围为 -k_w/2 到 k_w/2，k_h 和 k_w 分别为核的高度和宽度。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中
- 使用零填充处理边界条件（将图像边界外的值视为零）

## 示例

**示例 1：**

```
Input:
  image (5, 5) = [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0, 9.0, 10.0],
    [11.0, 12.0, 13.0, 14.0, 15.0],
    [16.0, 17.0, 18.0, 19.0, 20.0],
    [21.0, 22.0, 23.0, 24.0, 25.0]
  ]

  kernel (3, 3) = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]
  ]

Output:
  output (5, 5) = [
    [1.6875, 2.75, 3.5, 4.25, 3.5625],
    [4.75, 7.0, 8.0, 9.0, 7.25],
    [8.5, 12.0, 13.0, 14.0, 11.0],
    [12.25, 17.0, 18.0, 19.0, 14.75],
    [11.0625, 15.25, 16.0, 16.75, 12.9375]
  ]
```

**示例 2：**

```
Input:
  image (3, 3) = [
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0],
    [70.0, 80.0, 90.0]
  ]

  kernel (3, 3) = [
    [0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1],
    [0.1, 0.1, 0.1]
  ]

Output:
  output (3, 3) = [
    [13.0, 23.0, 19.0],
    [31.0, 50.0, 39.0],
    [31.0, 47.0, 37.0]
  ]
```

## 约束条件

- 1 ≤ `input_rows`, `input_cols` ≤ 4096
- 3 ≤ `kernel_rows`, `kernel_cols` ≤ 21
- `kernel_rows` 和 `kernel_cols` 均为奇数
- 所有核值为非负数且总和为 1.0（已归一化）
- 性能测试使用 `input_cols` = 512，`input_rows` = 512，`kernel_cols` = 7，`kernel_rows` = 7

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}
```
