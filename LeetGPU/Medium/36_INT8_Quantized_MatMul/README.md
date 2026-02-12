# INT8 Quantized MatMul

> LeetGPU: https://leetgpu.com/challenges/int8-quantized-matmul

## 难度

Medium

## 题目描述

实现一个 8 位有符号整数矩阵的量化矩阵乘法程序。给定维度为 M x K 的输入矩阵 `A` 和维度为 K x N 的输入矩阵 `B`、量化缩放因子 `scale_A`、`scale_B`、输出缩放因子 `scale_C`、零点 `zero_point_A`、`zero_point_B`、`zero_point_C`，计算：C_quant(i, j) = clamp(round((sum((A_{ik} - z_A) * (B_{kj} - z_B), k=0..K-1) * s_A * s_B) / s_C) + z_C, -128, 127)，其中 s_A = scale_A，z_A = zero_point_A，以此类推。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须以 `int8` 类型存储在输出矩阵 `C` 中
- 在 int32 中完成累加并使用 float32 进行缩放后，值必须四舍五入到最近的整数，加上 `zero_point_C`，并裁剪到 `[-128, 127]` 范围内

## 示例

**示例 1：**

```
Input:
     A = [[1, 2],
          [3, 4]]
     B = [[5, 6],
          [7, 8]]
     M = 2, N = 2, K = 2
     scale_A = 0.1, scale_B = 0.2, scale_C = 0.05
     zero_point_A = 0, zero_point_B = 0, zero_point_C = 0

     Output:
     C = [[19, 22],
          [43, 50]]
```

**示例 2：**

```
Input:
     A = [[1, 2]]
     B = [[3],
          [4]]
     M = 1, N = 1, K = 2
     scale_A = 1.0, scale_B = 1.0, scale_C = 1.0
     zero_point_A = 1, zero_point_B = 3, zero_point_C = 5

     Output:
     C = [[6]]
```

## 约束条件

- 1 ≤ `M`, `N`, `K` ≤ 4096
- `scale_A`、`scale_B`、`scale_C` 为正浮点数
- -128 ≤ `zero_point_A`, `zero_point_B`, `zero_point_C` ≤ 127
- 性能测试使用 `K` = 2,048，`M` = 8,192，`N` = 4,096

## 代码模板

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {}
```
