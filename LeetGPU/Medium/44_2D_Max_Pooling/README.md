# 2D Max Pooling

> LeetGPU: https://leetgpu.com/challenges/2d-max-pooling

## 难度

Medium

## 题目描述

实现用于图像/特征图下采样的 2D 最大池化操作。程序接收一个输入张量，使用指定的核大小、步幅和填充参数进行最大池化，生成输出张量。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  input = [[[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]]]
        kernel_size = 2
        stride = 1
        padding = 0
Output: output = [[[[5.0, 6.0],
                    [8.0, 9.0]]]]
```

**示例 2：**

```
Input:  input = [[[[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0],
                   [21.0, 22.0, 23.0, 24.0, 25.0]]]]
        kernel_size = 3
        stride = 1
        padding = 1
Output: output = [[[[7.0, 8.0, 9.0, 10.0, 10.0],
                    [12.0, 13.0, 14.0, 15.0, 15.0],
                    [17.0, 18.0, 19.0, 20.0, 20.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0]]]]
```

## 约束条件

- 1 ≤ N ≤ 100（批大小）
- 1 ≤ C ≤ 512（通道数）
- 1 ≤ H, W ≤ 1024（高度、宽度）
- 1 ≤ kernel_size ≤ 16
- 1 ≤ stride ≤ 16
- 0 ≤ padding ≤ 16
- 输入和输出张量使用 float32 精度
- 性能测试使用 `N` = 4, `kernel_size` = 3, `stride` = 2

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {}
```
