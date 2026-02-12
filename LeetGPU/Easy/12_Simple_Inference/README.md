# Simple Inference

> LeetGPU: https://leetgpu.com/challenges/simple-inference

## 难度

Easy

## 题目描述

对一个 PyTorch 模型执行推理。给定一个输入张量和一个已训练的 `torch.nn.Linear` 模型，计算前向传播并将结果存储在输出张量中。该模型执行线性变换：`output = input @ weight.T + bias`，其中 `weight` 的形状为 [output_size, input_size]，`bias` 的形状为 [output_size]。

## 实现要求

- 使用 PyTorch 的内置函数和操作
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 张量中
- 模型已加载完毕，可直接用于推理

## 示例

**示例 1：**

```
Input:  input = [[1.0, 2.0]]  (batch_size=1, input_size=2)
          model: Linear layer with weight=[[0.5, 1.0], [1.5, 0.5]], bias=[0.1, 0.2]
  Output: output = [[2.6, 2.7]]  (batch_size=1, output_size=2)
```

**示例 2：**

```
Input:  input = [[1.0], [2.0], [3.0]]  (batch_size=3, input_size=1)
          model: Linear layer with weight=[[2.0]], bias=[1.0]
  Output: output = [[3.0], [5.0], [7.0]]  (batch_size=3, output_size=1)
```

## 约束条件

- 1 ≤ `batch_size` ≤ 1,000
- 1 ≤ `input_size` ≤ 1,000
- 1 ≤ `output_size` ≤ 1,000
- -10.0 ≤ 输入值 ≤ 10.0
- 性能测试使用 `batch_size` = 1,000
