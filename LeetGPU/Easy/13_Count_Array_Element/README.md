# Count Array Element

> LeetGPU: https://leetgpu.com/challenges/count-array-element

## 难度

Easy

## 题目描述

编写一个 GPU 程序，统计一个 32 位整数数组中值等于整数 k 的元素个数。给定一个长度为 `N` 的输入数组 `input` 和一个整数 `k`，程序应统计数组中值等于 k 的元素数量。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input: [1, 2, 3, 4, 1], k = 1
Output: 2
```

**示例 2：**

```
Input: [5, 10, 5, 2], k = 11
Output: 0
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- 1 ≤ `input[i], k` ≤ 100,000
- 性能测试使用 `K` = 501,010，`N` = 100,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
```
