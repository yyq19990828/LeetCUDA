# FP16 Dot Product

> LeetGPU: https://leetgpu.com/challenges/fp16-dot-product

## 难度

Medium

## 题目描述

编写一个 GPU 程序，计算两个包含 16 位浮点数（FP16/`half`）的向量的点积。点积是两个向量对应元素乘积之和。数学上，两个长度为 n 的向量 A 和 B 的点积定义为：

A · B = sum(A_i * B_i, i=0..n-1) = A_0 * B_0 + A_1 * B_1 + ... + A_(n-1) * B_(n-1)

所有输入以 16 位浮点数（FP16/`half`）存储。为获得最佳精度，乘法过程中的累加应使用 FP32，最终结果再转换为 FP16。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 乘法过程中的累加应使用 FP32 以获得更高精度，最终结果再转换为 FP16
- 最终结果必须以 `half` 类型存储在输出变量中

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
- 性能测试使用 `N` = 100,000,000

## 代码模板

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {}
```
