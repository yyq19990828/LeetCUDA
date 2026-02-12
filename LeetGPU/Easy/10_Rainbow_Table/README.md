# Rainbow Table

> LeetGPU: https://leetgpu.com/challenges/rainbow-table

## 难度

Easy

## 题目描述

实现一个使用给定哈希函数对 32 位整数数组执行 `R` 轮并行哈希的程序。哈希应迭代应用 `R` 次（前一轮的输出作为下一轮的输入）。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在数组 `output` 中

## 示例

**示例 1：**

```
Input:  numbers = [123, 456, 789], R = 2
Output: hashes = [1636807824, 1273011621, 2193987222]
```

**示例 2：**

```
Input:  numbers = [0, 1, 2147483647], R = 3
Output: hashes = [96754810, 3571711400, 2006156166]
```

## 约束条件

- 1 ≤ `N` ≤ 10,000,000
- 1 ≤ `R` ≤ 100
- 0 ≤ `input[i]` ≤ 2147483647
- 性能测试使用 `N` = 5,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```
