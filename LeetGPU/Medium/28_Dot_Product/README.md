# Dot Product

> LeetGPU: https://leetgpu.com/challenges/dot-product

## 难度

Medium

## 题目描述

实现一个 GPU 程序，计算两个包含 32 位浮点数的向量的点积。点积是两个向量对应元素乘积的总和。数学上，两个长度为 n 的向量 A 和 B 的点积定义为：A · B = A_0 * B_0 + A_1 * B_1 + ... + A_{n-1} * B_{n-1}。

## 实现要求

- 仅使用 GPU 原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 output 变量中

## 示例

**示例 1：**

```
Input:  A = [1.0, 2.0, 3.0, 4.0]
               B = [5.0, 6.0, 7.0, 8.0]
       Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)
```

**示例 2：**

```
Input:  A = [0.5, 1.5, 2.5]
               B = [2.0, 3.0, 4.0]
       Output: result = 15.5  (0.5*2.0 + 1.5*3.0 + 2.5*4.0)
```

## 约束条件

- `A` 和 `B` 长度相同
- 1 ≤ `N` ≤ 100,000,000
- 性能测试使用 `N` = 5

## 代码模板

```cpp
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {}
```
