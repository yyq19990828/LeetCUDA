# Prefix Sum

> LeetGPU: https://leetgpu.com/challenges/prefix-sum

## 难度

Medium

## 题目描述

编写一个 GPU 程序，计算一个 32 位浮点数数组的前缀和（累积和）。对于输入数组 `[a, b, c, d, ...]`，前缀和为 `[a, a+b, a+b+c, a+b+c+d, ...]`。

## 实现要求

- 仅使用 GPU 原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input: [1.0, 2.0, 3.0, 4.0]
Output: [1.0, 3.0, 6.0, 10.0]
```

**示例 2：**

```
Input: [5.0, -2.0, 3.0, 1.0, -4.0]
Output: [5.0, 3.0, 6.0, 7.0, 3.0]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0
- 输出数组中的最大值不会超出 32 位浮点数的表示范围
- 性能测试使用 `N` = 250,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {}
```
