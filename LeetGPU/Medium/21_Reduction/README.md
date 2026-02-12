# Reduction

> LeetGPU: https://leetgpu.com/challenges/reduction

## 难度

Medium

## 题目描述

编写一个 GPU 程序，对 32 位浮点数数组执行并行归约操作以计算其总和。程序应接收一个输入数组，并输出一个包含所有元素之和的单一值。

## 实现要求

- 只允许使用 GPU 原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0
```

**示例 2：**

```
Input: [-2.5, 1.5, -1.0, 2.0]
Output: 0.0
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `input[i]` ≤ 1000.0
- 最终求和结果始终可以用 32 位浮点数表示
- 性能测试使用 `N` = 4,194,304

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {}
```
