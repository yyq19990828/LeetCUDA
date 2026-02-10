# Vector Addition

> LeetGPU: https://leetgpu.com/challenges/vector-addition

## 难度

Easy

## 题目描述

编写一个 GPU 程序，对两个包含 32 位浮点数的向量执行逐元素加法。程序接收两个等长的输入向量，输出一个包含它们逐元素之和的向量。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在向量 `C` 中

## 示例

**示例 1：**

```
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
```

**示例 2：**

```
Input:  A = [1.5, 1.5, 1.5]
        B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
```

## 约束条件

- 输入向量 `A` 和 `B` 长度相同
- 1 ≤ N ≤ 100,000,000
- 性能测试使用 N = 25,000,000

## 代码模板

```c++
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

## 解题思路

这是最基础的 CUDA 入门题。核心思想是让每个线程负责计算一个元素的加法：

1. **线程映射**：通过 `blockIdx.x * blockDim.x + threadIdx.x` 计算全局线程索引，将其映射到数组下标
2. **边界检查**：当 N 不是 blockDim 的整数倍时，最后一个 block 中会有多余线程，需要 `if (idx < N)` 防止越界访问
3. **核心计算**：`C[idx] = A[idx] + B[idx]`

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| threadsPerBlock | 256 | 每个 block 的线程数，通常取 128/256/512 |
| blocksPerGrid | ⌈N/256⌉ | block 数量，确保覆盖所有元素 |

### 优化方向

- **向量化访存**：使用 `float4` 一次读写 4 个 float，减少内存事务数量
- **Grid-stride loop**：当 N 很大时，使用循环让每个线程处理多个元素，减少 block 调度开销
