# Top-p Sampling

> LeetGPU: https://leetgpu.com/challenges/top-p-sampling

## 难度

Medium

## 题目描述

编写一个 GPU 程序，实现用于 LLM 推理的 top-p（nucleus）采样。Top-p 采样是一种文本生成技术，从累积概率超过阈值 p 的最小 token 集合中进行采样。这比纯 top-k 或贪心采样能更好地平衡随机性和质量。给定语言模型的 logits（未归一化的分数）：使用 softmax 将 logits 转换为概率；按概率降序排列 token；找到累积概率 ≥ p 的最小集合（即"核"）；将核内的概率重新归一化使其和为 1；使用提供的随机种子从核中采样一个 token。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 计算 softmax 时需确保数值稳定性

## 示例

**示例 1：**

```
Input:
  logits = [1.0, 2.0, 3.0, 0.5]
  p = 0.9
  seed = 42

Output:
  sampled_token = 2 or 1
  (tokens with highest probabilities, sampled randomly)
```

**示例 2：**

```
Input:
  logits = [10.0, 1.0, 1.0]
  p = 0.5
  seed = 123

Output:
  sampled_token = 0
  (single token dominates the probability mass)
```

## 约束条件

- 3 ≤ `vocab_size` ≤ 50,000
- -100.0 ≤ `logits[i]` ≤ 100.0
- 0.0 < `p` ≤ 1.0
- 0 ≤ `sampled_token` < vocab_size
- 性能测试使用 `vocab_size` = 50,000

## 代码模板

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {}
```
