# Count 2D Array Element

> LeetGPU: https://leetgpu.com/challenges/count-2d-array-element

## 难度

Easy

## 题目描述

编写一个 GPU 程序，统计一个 32 位整数二维数组中值等于整数 k 的元素个数。给定一个大小为 `N x M` 的输入二维数组 `input` 和一个整数 `k`，程序应统计二维数组中值等于 k 的元素数量。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 变量中

## 示例

**示例 1：**

```
Input: input [[1, 2, 3],
              [4, 5, 1]]
       k = 1
Output: output = 2
```

**示例 2：**

```
Input: input [[5, 10],
              [5, 2]]
       k = 1
Output: output = 0
```

## 约束条件

- 1 ≤ `N, M` ≤ 10,000
- 1 ≤ `input[i], k` ≤ 100
- 性能测试使用 `K` = 1，`M` = 10,000，`N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
```
