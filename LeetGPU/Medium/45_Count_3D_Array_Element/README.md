# Count 3D Array Element

> LeetGPU: https://leetgpu.com/challenges/count-3d-array-element

## 难度

Medium

## 题目描述

编写一个 GPU 程序，统计一个 32 位整数的三维数组中值为 p 的元素个数。给定一个大小为 `N x M x K` 的三维输入数组 `input` 和整数 `p`，程序需要统计数组中等于 p 的元素数量。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input: input [[[1, 2, 3],
               [4, 5, 1]],
              [[1, 1, 1],
               [2, 2, 2]]]
       N = 2, M = 2, K = 3
       p = 1
Output: output = 5
```

**示例 2：**

```
Input: input [[[5, 10],
               [5, 2],
               [2, 2]]]
       N = 1, M = 3, K = 2
       p = 1
Output: output = 0
```

## 约束条件

- 1 ≤ `N, M, K` ≤ 1,000
- 1 ≤ `input[i], p` ≤ 100
- 性能测试使用 `K` = 500, `M` = 500, `N` = 500

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {}
```
