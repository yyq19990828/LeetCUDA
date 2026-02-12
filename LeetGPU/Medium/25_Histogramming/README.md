# Histogramming

> LeetGPU: https://leetgpu.com/challenges/histogramming

## 难度

Medium

## 题目描述

编写一个 GPU 程序，计算 32 位整数数组的直方图。直方图应统计范围 `[0, num_bins)` 内每个整数值的出现次数。给定一个长度为 `N` 的输入数组 `input` 和 bin 数量 `num_bins`。结果应为一个长度为 `num_bins` 的整数数组，其中每个元素表示其对应索引值在输入数组中的出现次数。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `histogram` 数组中

## 示例

**示例：**

```
Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
Output: [2, 2, 1]

Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
Output: [0, 0, 0, 4, 0]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- 0 ≤ `input[i]` < `num_bins`
- 1 ≤ `num_bins` ≤ 1024
- 性能测试使用 `N` = 50,000,000，`num_bins` = 256

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {}
```
