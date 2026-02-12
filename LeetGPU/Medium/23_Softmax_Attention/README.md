# Softmax Attention

> LeetGPU: https://leetgpu.com/challenges/softmax-attention

## 难度

Medium

## 题目描述

编写一个 GPU 程序，对给定的矩阵集合计算 softmax 注意力操作。给定大小为 `M*d` 的查询矩阵 `Q`、大小为 `N*d` 的键矩阵 `K` 和大小为 `N*d` 的值矩阵 `V`，程序应使用以下公式计算输出矩阵：Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V，其中 softmax 函数按行应用。

## 实现要求

- 只允许使用 GPU 原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在输出矩阵 `output` 中

## 示例

**示例 1：**

```
Input:
Q (2x4):
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0]]

K (3x4):
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0]]

V (3x4):
[[1.0, 2.0, 3.0, 4.0],
 [5.0, 6.0, 7.0, 8.0],
 [9.0, 10.0, 11.0, 12.0]]

Output (2x4):
[[4.29, 5.29, 6.29, 7.29],
 [5.00, 6.00, 7.00, 8.00]]
```

**示例 2：**

```
Input:
Q (1x2):
[[1.0, 2.0]]

K (2x2):
[[1.0, 0.0],
 [0.0, 1.0]]

V (2x2):
[[3.0, 4.0],
 [5.0, 6.0]]

Output (1x2):
[[4.34, 5.34]]
```

## 约束条件

- 矩阵 `Q` 的大小为 `M*d`，矩阵 `K` 和 `V` 的大小为 `N*d`
- 1 ≤ `M`, `N` ≤ 100,000
- 1 ≤ `d` ≤ 128
- 性能测试使用 `M` = 512，`N` = 256

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {}
```
