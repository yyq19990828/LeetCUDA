# Categorical Cross Entropy Loss

> LeetGPU: https://leetgpu.com/challenges/categorical-cross-entropy-loss

## 难度

Medium

## 题目描述

实现一个 GPU 程序，计算一批预测值的分类交叉熵损失。给定一个大小为 N x C 的预测 logits 矩阵 Z 和一个大小为 N 的真实类别标签向量 `true_labels`，计算该批次的平均交叉熵损失。对于具有 logits z_j = [z_{j1}, ..., z_{jC}] 和真实标签 y_j 的单个样本 j，其损失使用数值稳定公式计算为：Loss_j = log(sum(exp(z_{jk}), k=1..C)) - z_{j, y_j}。最终存储在 `loss` 变量中的输出应为 N 个样本的平均损失：L = (1/N) * sum(Loss_j, j=1..N)。输入参数为 `logits`、`true_labels`、`N`（样本数）和 `C`（类别数）。结果应存储在 `loss` 中（指向单个 float 的指针）。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果（平均损失）必须存储在 `loss` 中

## 示例

**示例 1：**

```
Input:  N = 2, C = 3
        logits = [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]
        true_labels = [1, 1]
Output: loss = [0.3548926]
```

**示例 2：**

```
Input:  N = 3, C = 4
        logits = [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]
        true_labels = [3, 0, 1]
Output: loss = [0.98820376]
```

## 约束条件

- 1 ≤ `N` ≤ 10,000
- 2 ≤ `C` ≤ 1,000
- -10.0 ≤ `logits[i, j]` ≤ 10.0
- 0 ≤ `true_labels[i]` ≤ `C`
- 性能测试使用 `N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {}
```
