# Reverse Array

> LeetGPU: https://leetgpu.com/challenges/reverse-array

## 难度

Easy

## 题目描述

实现一个将 32 位浮点数数组原地反转的程序。程序应对 `input` 进行原地反转。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储回 `input` 中

## 示例

**示例 1：**

```
Input: [1.0, 2.0, 3.0, 4.0]
Output: [4.0, 3.0, 2.0, 1.0]
```

**示例 2：**

```
Input: [1.5, 2.5, 3.5]
Output: [3.5, 2.5, 1.5]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- 性能测试使用 `N` = 25,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```
