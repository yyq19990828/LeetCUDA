# Ordinary Least Squares

> LeetGPU: https://leetgpu.com/challenges/ordinary-least-squares

## 难度

Medium

## 题目描述

在 GPU 上求解普通最小二乘法（OLS）回归问题。给定一个大小为 n_samples x n_features 的特征矩阵 X 和一个大小为 n_samples 的目标向量 y，计算使残差平方和最小的系数向量 beta：min_beta ||X * beta - y||^2。OLS 的闭式解为：beta = (X^T * X)^(-1) * X^T * y。

## 实现要求

- 不允许使用外部库。
- `solve` 函数签名不可修改。
- 最终系数必须存储在 `beta` 向量中。
- 假设特征矩阵 X 是满秩的（即 X^T * X 可逆）。

## 示例

**示例：**

```
Input: X (samples × features):
[[-0.23, -0.23, 1.52],
 [0.77, -0.47, 1.58],
 [-0.14, 0.65, 0.5],
 [-1.91, -1.72, 0.24],
 [-0.46, -0.47, 0.54]]

y:
[83.01, 93.4, 47.33, -62.22, 13.06]

Output: beta:
[13.97, 29.12, 61.05]
```

## 约束条件

- 1 ≤ `n_samples` ≤ 100,000
- 1 ≤ `n_features` ≤ 1,000
- `n_samples` ≥ `n_features`
- -1000.0 ≤ `X` 和 `y` 中的值 ≤ 1000.0
- 使用绝对容差 1e-2 和相对容差 1e-2 进行测试
- 性能测试使用 `n_features` = 32, `n_samples` = 32

## 代码模板

```cpp
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}
```
