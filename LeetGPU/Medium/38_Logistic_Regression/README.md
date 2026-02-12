# Logistic Regression

> LeetGPU: https://leetgpu.com/challenges/logistic-regression

## 难度

Medium

## 题目描述

在 GPU 上求解逻辑回归问题。给定一个大小为 n_samples x n_features 的特征矩阵 X 和一个大小为 n_samples 的二元目标向量 y（仅包含 0 和 1），计算使对数似然最大化的系数向量 beta：max_beta sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]，其中 p_i = sigma(X_i^T * beta)，sigma(z) = 1 / (1 + e^(-z)) 为 sigmoid 函数。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终系数必须存储在 `beta` 向量中
- 目标向量 `y` 仅包含二元值（0 和 1）

## 示例

**示例：**

```
Input: X (samples × features):
[[2.0, 1.0],
 [1.0, 2.0],
 [3.0, 3.0],
 [1.5, 2.5],
 [-1.0, -2.0],
 [-2.0, -1.0],
 [-1.5, -2.5],
 [-3.0, -3.0]]

y:
[1, 1, 1, 0, 0, 0, 1, 0]

Output: beta:
[2.26, -1.29]
```

## 约束条件

- 1 ≤ `n_samples` ≤ 100,000
- 1 ≤ `n_features` ≤ 1,000
- `n_samples` ≥ `n_features`
- -10.0 ≤ `X` 中的值 ≤ 10.0
- `y` 仅包含二元值：0 或 1
- 使用绝对容差 1e-2 和相对容差 1e-2 进行测试
- 性能测试使用 `n_features` = 8, `n_samples` = 16

## 代码模板

```cpp
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}
```
