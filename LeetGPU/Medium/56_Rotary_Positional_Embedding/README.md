# Rotary Positional Embedding

> LeetGPU: https://leetgpu.com/challenges/rotary-positional-embedding

## 难度

Medium

## 题目描述

编写一个 GPU 程序，为一批查询向量计算旋转位置编码（RoPE）。RoPE 是一种在 transformer 模型中编码位置信息的方法，通过使用预计算的余弦和正弦分量来旋转查询和键向量。数学上，给定查询向量 x 及对应的余弦和正弦向量，操作定义为：

RoPE(x) = x * cos + rotate_half(x) * sin

其中 * 表示逐元素乘法。`rotate_half(x)` 操作将向量的前半部分和后半部分交换，并对前半部分取反。对于维度为 d 的向量：

rotate_half([x_1, ..., x_(d/2), x_(d/2+1), ..., x_d]) = [-x_(d/2+1), ..., -x_d, x_1, ..., x_(d/2)]

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 输入张量 `Q`、`cos` 和 `sin` 的形状为 `(M, D)`，其中 `M` 是 token 数量，`D` 是头维度
- `D`（头维度）保证为偶数
- 最终结果必须存储在形状相同的 `(M, D)` 输出变量中

## 示例

**示例 1：**

```
Input:  Q   = [[1.0, 2.0, 3.0, 4.0],
               [1.0, 1.0, 1.0, 1.0]]
        Cos = [[1.0, 1.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 0.0]]
        Sin = [[0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0]]
Output: result = [[1.0, 2.0, 3.0, 4.0],
                  [-1.0, -1.0, 1.0, 1.0]]
        (Row 0 is identity via Cos; Row 1 is rotated via Sin)
```

## 约束条件

- `Q`、`cos` 和 `sin` 具有相同的维度
- `D` % 2 == 0
- 1 ≤ `M`, `D` ≤ 10,000
- 性能测试使用 `D` = 128, `M` = 1,048,576

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {}
```
