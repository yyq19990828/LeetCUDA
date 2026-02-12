# MoE Top-K Gating

> LeetGPU: https://leetgpu.com/challenges/moe-top-k-gating

## 难度

Medium

## 题目描述

编写一个 GPU 程序，为混合专家（MoE）模型执行 Top-K 门控。给定一个形状为 `[M, E]` 的 logit 矩阵，其中 M 是 token 数量，E 是专家数量，找出每行中最大的 k 个值，提取其索引，并应用 softmax 得到混合权重。对于每行 i，操作计算如下：

indices_i = argsort(logits_i)[-k:]

vals_i = logits_i[indices_i]

weights_i = Softmax(vals_i)

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `topk_weights` 和 `topk_indices` 数组中

## 示例

**示例 1：**

```
Input:
  logits = [[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]]
  M = 2, E = 4, k = 2

Output:
  topk_weights = [[0.2689, 0.7311],
                  [0.7311, 0.2689]]
  topk_indices = [[2, 3],
                  [0, 1]]

Explanation:
Row 0: Top-2 values are 3.0 and 4.0 at indices 2 and 3.
       Softmax([3.0, 4.0]) = [0.2689, 0.7311]
Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
       Softmax([4.0, 3.0]) = [0.7311, 0.2689]
```

## 约束条件

- 1 ≤ `M` ≤ 10,000（token 数量）
- 1 ≤ `E` ≤ 256（专家数量）
- 1 ≤ `k` ≤ `E`（top-k 选择，通常 k=2）
- 所有张量存储在 GPU 上
- Logits 为 32 位浮点数
- 索引为 32 位整数
- 性能测试使用 `M` = 1,024, `k` = 2

## 代码模板

```cpp
#include <cuda_runtime.h>

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {}
```
