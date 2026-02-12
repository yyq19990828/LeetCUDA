# Max Subarray Sum

> LeetGPU: https://leetgpu.com/challenges/max-subarray-sum

## 难度

Medium

## 题目描述

实现一个程序，计算长度恰好为 `window_size` 的连续子数组的最大和。给定一个长度为 `N` 的 32 位有符号整数数组 `input`，以及一个整数 `window_size`。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input:  input = [1, 2, 4, 2, 3], window_size = 2
Output: output = 6
```

**示例 2：**

```
Input:  input = [-1, -4, -2, 1], window_size = 3
Output: output = -5
```

## 约束条件

- 1 ≤ `N` ≤ 50,000
- -10 ≤ `input[i]` ≤ 10
- 1 ≤ `window_size` ≤ `N`
- 性能测试使用 `N` = 50,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {}
```
