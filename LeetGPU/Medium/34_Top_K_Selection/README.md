# Top K Selection

> LeetGPU: https://leetgpu.com/challenges/top-k-selection

## 难度

Medium

## 题目描述

实现一个 GPU 程序，给定一个长度为 `N` 的 32 位浮点数一维数组 `input`，选出最大的 `k` 个元素，并按降序写入长度为 `k` 的 `output` 数组中。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input:
  input = [1.0, 5.0, 3.0, 2.0, 4.0]
  N = 5
  k = 3

  Output:
  output = [5.0, 4.0, 3.0]
```

**示例 2：**

```
Input:
  input = [7.2, -1.0, 3.3, 8.8, 2.2]
  N = 5
  k = 2

  Output:
  output = [8.8, 7.2]
```

## 约束条件

- 1 ≤ N ≤ 100,000,000
- 1 ≤ k ≤ N
- `input` 中的所有值均为 32 位浮点数
- 性能测试使用 `N` = 50,000,000，`k` = 100

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {}
```
