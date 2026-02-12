# Sliding Window Self-Attention

> LeetGPU: https://leetgpu.com/challenges/sliding-window-self-attention

## 难度

Hard

## 题目描述

实现滑动窗口自注意力。在介绍滑动窗口版本之前，我们先回顾标准自注意力。

1. 标准 Softmax 注意力

给定查询矩阵 `Q`、键矩阵 `K` 和值矩阵 `V`，每个位置 `i` 使用 softmax 加权和关注所有位置 `j`：

score_(i,j) = Q_i · K_j / sqrt(d)

output_i = sum_j(softmax(score_(i,*))_j * V_j)，其中 j 从 1 到 M

换句话说，每个查询计算与所有键的相似度，应用 softmax 得到注意力权重，然后计算值的加权和。

2. 滑动窗口自注意力

滑动窗口注意力通过限制每个查询只关注其位置周围的局部窗口来修改标准注意力。对于每个位置 `i`，只考虑大小为 `window_size` 的窗口内的键和值（位置 [i-window_size, ..., i+window_size]）。计算 Q_i 与窗口内键之间的相似度分数：

score_(i,j) = Q_i · K_j / sqrt(d)

对这些局部分数应用 softmax 得到注意力权重。使用权重计算同一窗口内值的加权平均：

output_i = sum(softmax(score_(i,*))_j * V_j)，其中 j 属于 [i-window_size, i+window_size]

简而言之，每个查询只关注其附近的邻居。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在输出矩阵 `output` 中

## 示例

**示例 1：**

```
Input: `Q` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `K` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `V` (2×4): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix} \] `window_size`: 1 Output: `output` (2×4): \[ \begin{bmatrix} 2.5101628 & 3.5101628 & 4.510163 & 5.510163 \\ 3.4898374 & 4.4898376 & 5.4898376 & 6.489837 \end{bmatrix} \]
```

**示例 2：**

```
Input: `Q` (2×3): \[ \begin{bmatrix} 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 \end{bmatrix} \] `K` (2×3): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 \end{bmatrix} \] `V` (2×3): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 \\ 5.0 & 6.0 & 7.0 \end{bmatrix} \] `window_size`: 1 Output: `output` (2×3): \[ \begin{bmatrix} 3.0 & 4.0 & 5.0 \\ 3.5618298 & 4.56183 & 5.5618296 \end{bmatrix} \]
```

## 约束条件

- 矩阵 `Q`、`K` 和 `V` 的大小均为 `M×d`
- 1 ≤ `M` ≤ 10000
- 1 ≤ `d` ≤ 128
- 1 ≤ `window_size` ≤ 32
- `Q`、`K` 和 `V` 中的所有元素从 `[-100.0, 100.0]` 范围内采样
- 所有矩阵的数据类型为 `float32`
- 性能测试使用 `M` = 5,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {}
```
