# Batch Normalization

> LeetGPU: https://leetgpu.com/challenges/batch-normalization

## 难度

Medium

## 题目描述

实现 2D 输入张量的批量归一化前向传播。给定一个形状为 [N, C] 的输入张量，其中 N 为批大小，C 为特征数量，使用可学习的缩放参数 `gamma` 和偏移参数 `beta` 计算归一化输出。对于每个特征通道 j，批量归一化的计算过程如下：mu_j = (1/N) * sum(x_{i,j})，sigma_j^2 = (1/N) * sum((x_{i,j} - mu_j)^2)，x_hat_{i,j} = (x_{i,j} - mu_j) / sqrt(sigma_j^2 + epsilon)，y_{i,j} = gamma_j * x_hat_{i,j} + beta_j。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
        gamma = [1.0, 1.0]
        beta = [0.0, 0.0]
        eps = 1e-5
Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]
```

**示例 2：**

```
Input:  input = [[0.0, 1.0], [2.0, 3.0]]  (N=2, C=2)
        gamma = [2.0, 0.5]
        beta = [1.0, -1.0]
        eps = 1e-5
Output: output = [[-1.0, -1.5], [3.0, -0.5]]
```

## 约束条件

- 1 ≤ `N` ≤ 10,000
- 1 ≤ `C` ≤ 1,024
- `eps` = 1e-5
- -100.0 ≤ 输入值 ≤ 100.0
- 0.1 ≤ gamma 值 ≤ 10.0
- -10.0 ≤ beta 值 ≤ 10.0
- 性能测试使用 `N` = 5,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {}
```
