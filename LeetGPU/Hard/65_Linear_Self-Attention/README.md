# Linear Self-Attention

> LeetGPU: https://leetgpu.com/challenges/linear-self-attention

## 难度

Hard

## 题目描述

实现线性注意力，遵循论文"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"中描述的方法。给定大小为 `M×d` 的查询矩阵 `Q`、大小为 `M×d` 的键矩阵 `K` 和大小为 `M×d` 的值矩阵 `V`，你的程序应使用以下公式计算输出矩阵：

LinearAttention(Q, K, V) = phi(Q) * (phi(K)^T * V) / (phi(Q) * sum_j(phi(K_j)))

其中 phi(x) 是逐元素应用的特征映射函数，例如：

phi(x) = ELU(x) + 1 = { x + 1, 当 x > 0; e^x, 当 x <= 0 }

所有矩阵 `Q`、`K`、`V` 和 `output` 的类型为 `float32`，`M` 和 `d` 的类型为 `int32`。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在输出矩阵 `output` 中

## 示例

**示例 1：**

```
Input: `Q` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `K` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `V` (2×4): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix} \] Output: `output` (2×4): \[ \begin{bmatrix} 2.8461537 & 3.8461537 & 4.8461537 & 5.8461537 \\ 3.1538463 & 4.1538463 & 5.1538463 & 6.1538463 \end{bmatrix} \]
```

**示例 2：**

```
Input: `Q` (2×2): \[ \begin{bmatrix} 0.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix} \] `K` (2×2): \[ \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \] `V` (2×2): \[ \begin{bmatrix} 3.0 & 4.0 \\ 5.0 & 6.0 \end{bmatrix} \] Output: `output` (2×2): \[ \begin{bmatrix} 4.0 & 5.0 \\ 4.0 & 5.0 \end{bmatrix} \]
```

## 约束条件

- 矩阵 `Q`、`K` 和 `V` 的大小均为 `M×d`
- 1 ≤ `M` ≤ 10000
- 1 ≤ `d` ≤ 128
- `Q`、`K` 和 `V` 中的所有元素从 `[-100.0, 100.0]` 范围内采样
- 所有矩阵的数据类型为 `float32`
- 性能测试使用 `M` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {

}
```
