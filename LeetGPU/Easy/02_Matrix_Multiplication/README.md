# Matrix Multiplication

> LeetGPU: https://leetgpu.com/challenges/matrix-multiplication

## 难度

Easy

## 题目描述

编写一个 GPU 程序，将两个 32 位浮点数矩阵相乘。给定维度为 M × N 的矩阵 A 和维度为 N × K 的矩阵 B，计算乘积矩阵 C = A × B，结果矩阵 C 的维度为 M × K。所有矩阵以**行优先（row-major）**格式存储。

## 实现要求

- 仅可使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在矩阵 `C` 中

## 示例

**示例 1：**

```
Input:
Matrix A (2 × 2):      Matrix B (2 × 2):
| 1.0  2.0 |           | 5.0  6.0 |
| 3.0  4.0 |           | 7.0  8.0 |

Output:
Matrix C (2 × 2):
| 19.0  22.0 |
| 43.0  50.0 |
```

**示例 2：**

```
Input:
Matrix A (1 × 3):      Matrix B (3 × 1):
| 1.0  2.0  3.0 |      | 4.0 |
                        | 5.0 |
                        | 6.0 |

Output:
Matrix C (1 × 1):
| 32.0 |
```

## 约束条件

- 1 ≤ M, N, K ≤ 8192
- 性能测试使用 M = 8192, N = 6144, K = 4096

## 代码模板

```c++
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

## 解题思路

每个线程负责计算输出矩阵 C 中的一个元素 `C[row][col]`：

1. **2D 线程映射**：使用 `dim3` 定义 2D block/grid，`row = blockIdx.y * blockDim.y + threadIdx.y`，`col = blockIdx.x * blockDim.x + threadIdx.x`
2. **内积计算**：对 A 的第 row 行与 B 的第 col 列做点积，累加 N 次乘法
3. **行优先寻址**：`A[row * N + i]`、`B[i * K + col]`、`C[row * K + col]`

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| threadsPerBlock | (16, 16) | 每个 block 256 个线程，2D 布局 |
| blocksPerGrid | (⌈K/16⌉, ⌈M/16⌉) | 覆盖输出矩阵所有元素 |

### 优化方向

- **Shared Memory Tiling**：将 A、B 的子块加载到共享内存，减少全局内存访问次数，从 O(N) 降到 O(N/TILE_SIZE)
- **增大 Tile 尺寸**：使用 32×32 或更大的 tile，提高计算访存比
- **寄存器分块**：每个线程计算多个输出元素（如 TM×TN 的子块），提升指令级并行度
- **向量化访存**：使用 `float4` 一次加载 128 bit，充分利用内存带宽
