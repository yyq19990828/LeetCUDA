# Matrix Transpose

> LeetGPU: https://leetgpu.com/challenges/matrix-transpose

## 难度

Easy

## 题目描述

编写一个 GPU 程序，对一个 32 位浮点数矩阵执行转置操作。矩阵转置即交换行与列。给定维度为 rows × cols 的矩阵 A，其转置 A^T 的维度为 cols × rows。所有矩阵以**行优先（row-major）**格式存储。

## 实现要求

- 仅可使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `output` 矩阵中

## 示例

**示例 1：**

```
Input: 2×3 matrix
| 1.0  2.0  3.0 |
| 4.0  5.0  6.0 |

Output: 3×2 matrix
| 1.0  4.0 |
| 2.0  5.0 |
| 3.0  6.0 |
```

**示例 2：**

```
Input: 3×1 matrix
| 1.0 |
| 2.0 |
| 3.0 |

Output: 1×3 matrix
| 1.0  2.0  3.0 |
```

## 约束条件

- 1 ≤ rows, cols ≤ 8192
- 输入矩阵维度：rows × cols
- 输出矩阵维度：cols × rows
- 性能测试使用 cols = 6,000, rows = 7,000

## 代码模板

```c++
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {}

// input, output are device pointers (i.e. pointers to memory on the GPU).
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

## 解题思路

矩阵转置的核心挑战是**内存访问模式**：读和写必然有一个是非合并的。

### Naive 方案

每个线程读 `input[row * cols + col]`，写 `output[col * rows + row]`。读是合并的（同一 warp 连续列），但写是跨步的（步长为 rows），性能差。

### Shared Memory 优化

1. **按 tile 读入**：一个 block 处理一个 TILE×TILE 的子块，从 `input` 合并读入共享内存
2. **共享内存中转置**：读 `tile[threadIdx.y][threadIdx.x]`，写时交换为 `tile[threadIdx.x][threadIdx.y]`
3. **合并写出**：写到 `output` 时也是连续地址，实现读写双向合并

### Bank Conflict 避免

共享内存声明为 `tile[TILE][TILE+1]`，多加一列 padding，避免同一 warp 的线程在转置读取时访问同一 bank。

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_SIZE | 32 | tile 边长，匹配 warp 大小 |
| threadsPerBlock | (32, 8) | 每线程处理 4 行，提高 occupancy |
