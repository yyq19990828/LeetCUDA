# Multi-Head Attention

> LeetGPU: https://leetgpu.com/challenges/multi-head-attention

## 难度

Hard

## 题目描述

实现多头自注意力程序。给定三个大小为 N × d_model 的输入矩阵 Q（查询）、K（键）和 V（值），计算：

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)

其中每个头计算：

head_i = softmax(Q_i * K_i^T / sqrt(d_k)) * V_i

其中 d_k = d_model / h，Q_i、K_i、V_i 是输入矩阵中第 i 个头的分区。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 数组中

## 示例

**示例 1：**

```
Input: \[ \begin{align*} N &= 2, \quad d_{\text{model}} = 4, \quad h = 2 \\[1em] Q &= \begin{bmatrix} 1.0 & 0.0 & 2.0 & 3.0 \\ 4.0 & 5.0 & 6.0 & 7.0 \end{bmatrix} \\[1em] K &= \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix} \\[1em] V &= \begin{bmatrix} 0.5 & 1.0 & 1.5 & 2.0 \\ 2.5 & 3.0 & 3.5 & 4.0 \end{bmatrix} \end{align*} \] Output: \[ \begin{bmatrix} 2.39 & 2.89 & 3.50 & 4.00 \\ 2.50 & 3.00 & 3.50 & 4.00 \end{bmatrix} \]
```

**示例 2：**

```
Input: \[ \begin{align*} N &= 1, \quad d_{\text{model}} = 2, \quad h = 1 \\[1em] Q &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em] K &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em] V &= \begin{bmatrix} 2.0 & 3.0 \end{bmatrix} \end{align*} \] Output: \[ \begin{bmatrix} 2.0 & 3.0 \end{bmatrix} \]
```

## 约束条件

- `1 ≤ N ≤ 10000`
- `2 ≤ d_model ≤ 1024`
- `1 ≤ h ≤ d_model`
- `d_model % h == 0`
- `-10.0 ≤ 值 ≤ 10.0`
- 性能测试使用 `N` = 1,024, `d_model` = 1,024

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {}
```
