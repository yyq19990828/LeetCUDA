# RMS Normalization

> LeetGPU: https://leetgpu.com/challenges/rms-normalization

## 难度

Medium

## 题目描述

实现一维输入向量的 RMS Normalization 前向传播。给定一个形状为 [N] 的输入张量，其中 N 是元素数量，使用标量缩放参数（`gamma`）和偏移参数（`beta`）计算归一化输出。RMS Normalization 的计算公式如下：

rms = sqrt((1/N) * sum(x_i^2) + epsilon)

x_hat_i = x_i / rms

y_i = gamma * x_hat_i + beta

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中

## 示例

**示例 1：**

```
Input:  input = [1.0, 2.0, 3.0, 4.0]  (N=4)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.36514813, 0.73029625, 1.0954444, 1.4605925 ]
```

**示例 2：**

```
Input:  input = [1.0, 2.0, 3.0]  (N=3)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.46290955, 0.9258191, 1.3887286]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000
- `eps` = 1e-5
- -100.0 ≤ 输入值 ≤ 100.0
- 0.1 ≤ gamma ≤ 10.0
- -10.0 ≤ beta ≤ 10.0
- 性能测试使用 `N` = 100,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {}
```
