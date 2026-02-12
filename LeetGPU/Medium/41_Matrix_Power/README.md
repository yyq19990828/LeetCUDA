# Matrix Power

> LeetGPU: https://leetgpu.com/challenges/matrix-power

## 难度

Medium

## 题目描述

实现一个 GPU 程序，将一个大小为 N x N 的方阵 A 进行整数次幂运算。`solve` 函数接收一个展平的输入矩阵 `input`（按行优先顺序存储）、一个相同大小的空输出矩阵 `output`、维度 `N` 和指数 `P`。你需要计算 output = A^P，其中矩阵乘法为标准的 32 位浮点数稠密矩阵乘法。

## 实现要求

- 不允许使用外部库。
- `solve` 函数签名不可修改。
- 最终结果必须按行优先顺序写入 `output` 数组。

## 示例

**示例 1：**

```
Input:
    input  = [[1.0, 2.0],
              [3.0, 4.0]]
    N      = 2
    P      = 3
  Output:
    output = [[37.0, 54.0],
              [81.0, 118.0]]
```

**示例 2：**

```
Input:
    input  = [[1.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 0.0]]
    N      = 3
    P      = 2
  Output:
    output = [[7.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 6.0]]
```

## 约束条件

- 1 ≤ N ≤ 1024
- 1 ≤ P ≤ 20
- `input` 的元素满足 -10.0 ≤ A_ij ≤ 10.0
- 性能测试使用 `N` = 512

## 代码模板

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {}
```
