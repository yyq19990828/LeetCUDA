# Subarray Sum

> LeetGPU: https://leetgpu.com/challenges/subarray-sum

## 难度

Medium

## 题目描述

实现一个程序，计算 32 位整数子数组的和。给定一个长度为 `N` 的输入数组 `input`，以及两个索引 `S` 和 `E`。`S` 和 `E` 是包含边界的、从 0 开始的起止索引——计算 `input[S..E]` 的和。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6
```

**示例 2：**

```
Input: input = [1, 2, 3, 4], S = 0, E = 3
Output: output = 10
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i]` ≤ 10
- 0 ≤ `S` ≤ `E` ≤ `N - 1`
- 性能测试使用 `N` = 100,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {}
```
