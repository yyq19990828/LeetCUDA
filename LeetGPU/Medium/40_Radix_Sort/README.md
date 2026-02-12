# Radix Sort

> LeetGPU: https://leetgpu.com/challenges/radix-sort

## 难度

Medium

## 题目描述

在 GPU 上实现基数排序算法，对一个 32 位无符号整数数组进行排序。程序接收一个无符号整数的输入数组，使用基数排序算法将其按升序排列。`input` 参数包含未排序的数组，排序结果应存储在 `output` 数组中。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终排序结果必须存储在 `output` 数组中
- 使用基数排序算法（不能使用其他排序算法）
- 按升序排列

## 示例

**示例 1：**

```
Input:  [170, 45, 75, 90, 2, 802, 24, 66]
  Output: [2, 24, 45, 66, 75, 90, 170, 802]
```

**示例 2：**

```
Input:  [1, 4, 1, 3, 555, 1000, 2]
  Output: [1, 1, 2, 3, 4, 555, 1000]
```

## 约束条件

- `1 ≤ N ≤ 100,000,000`
- `0 ≤ input[i] ≤ 4,294,967,295`（32 位无符号整数）
- 性能测试使用 `N` = 50,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {}
```
