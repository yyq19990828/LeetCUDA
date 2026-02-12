# Weight Dequantization

> LeetGPU: https://leetgpu.com/challenges/weight-dequantization

## 难度

Medium

## 题目描述

编写一个 GPU 程序，在 GPU 上对权重矩阵进行"反量化"。给定一个形状为 `[M, N]` 的输入矩阵 `X`（包含量化值）和一个形状为 `[ceil(M/T), ceil(N/T)]` 的缩放矩阵 `S`，其中 `T` 是分块大小。对于每个元素 X_(i,j)，其对应的缩放因子为 S_(row, col)，其中 row = floor(i / T)，col = floor(j / T)。输出 Y_(i,j) 的计算方式为：

Y_(i,j) = X_(i,j) × S_(row, col)

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在输出缓冲区 `Y` 中

## 示例

**示例 1：**

```
Input:
M = 4, N = 4, TILE_SIZE = 2
X = [
  [10, 10,  5,  5],
  [10, 10,  5,  5],
  [ 2,  2,  8,  8],
  [ 2,  2,  8,  8]
]
S = [
  [0.5, 2.0],
  [4.0, 0.25]
]

Output:
Y = [
  [ 5.0,  5.0, 10.0, 10.0],
  [ 5.0,  5.0, 10.0, 10.0],
  [ 8.0,  8.0,  2.0,  2.0],
  [ 8.0,  8.0,  2.0,  2.0]
]
Explanation:
Tile (0,0) of X is multiplied by S[0,0] (0.5).
Tile (0,1) of X is multiplied by S[0,1] (2.0).
Tile (1,0) is multiplied by S[1,0] (4.0).
Tile (1,1) is multiplied by S[1,1] (0.25).
```

## 约束条件

- 1 ≤ `M`, `N` ≤ 8192
- `TILE_SIZE` 属于 {16, 32, 64, 128}
- 性能测试使用 `M` = 8,192, `N` = 8,192

## 代码模板

```cpp
#include <cuda_runtime.h>

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {}
```
