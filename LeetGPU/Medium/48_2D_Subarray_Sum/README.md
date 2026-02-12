# 2D Subarray Sum

> LeetGPU: https://leetgpu.com/challenges/2d-subarray-sum

## 难度

Medium

## 题目描述

实现一个程序，计算 32 位整数二维子数组的和。给定一个大小为 `N x M` 的二维输入数组 `input`，以及两个行索引 `S_ROW` 和 `E_ROW`、两个列索引 `S_COL` 和 `E_COL`。`S_ROW`、`E_ROW`、`S_COL` 和 `E_COL` 是包含边界的、从 0 开始的起止索引——计算 `input[S_ROW..E_ROW][S_COL..E_COL]` 的和。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input:  input = [[1, 2, 3],
                 [4, 5, 1]]

        S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2
Output: output = 11
```

**示例 2：**

```
Input:  input = [[5, 10],
                 [5, 2]]
        S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 1
Output: output = 10
```

## 约束条件

- 1 ≤ `N, M` ≤ 10,000
- 1 ≤ `input[i]` ≤ 10
- 0 ≤ `S_ROW` ≤ `E_ROW` ≤ `N - 1`
- 0 ≤ `S_COL` ≤ `E_COL` ≤ `M - 1`
- 性能测试使用 `M` = 10,000, `N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {}
```
