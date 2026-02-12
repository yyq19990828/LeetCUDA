# Causal Self-Attention

> LeetGPU: https://leetgpu.com/challenges/causal-self-attention

## 难度

Hard

## 题目描述

实现因果（掩码）自注意力。给定大小为 `M×d` 的查询矩阵 `Q`、大小为 `M×d` 的键矩阵 `K` 和大小为 `M×d` 的值矩阵 `V`，你的程序应使用以下公式计算输出矩阵：

Attention_causal(Q, K, V) = softmax(masked(Q * K^T / sqrt(d))) * V

其中 `mask` 是因果掩码，将所有对应于当前查询之后的键的位置设置为负无穷大。即，对于查询 `i` 和键 `j`：

当 j <= i 时，masked(a_(ij)) = a_(ij)；当 j > i 时，masked(a_(ij)) = -inf

softmax 函数按行应用。`Q`、`K`、`V` 和 `output` 的数据类型均为 `float32`；`M` 和 `d` 的数据类型为 `int32`。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在输出矩阵 `output` 中

## 示例

**示例 1：**

```
Input: `Q` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `K` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `V` (2×4): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \end{bmatrix} \] Output: `output` (2×4): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 3.4898374 & 4.4898374 & 5.4898374 & 6.4898374 \end{bmatrix} \]
```

**示例 2：**

```
Input: `Q` (2×2): \[ \begin{bmatrix} 0.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix} \] `K` (2×2): \[ \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \] `V` (2×2): \[ \begin{bmatrix} 3.0 & 4.0 \\ 5.0 & 6.0 \end{bmatrix} \] Output: `output` (2×2): \[ \begin{bmatrix} 3.0 & 4.0 \\ 5.0 & 6.0 \end{bmatrix} \]
```

## 约束条件

- 矩阵 `Q`、`K` 和 `V` 的大小均为 `M×d`
- 1 ≤ `M` ≤ 10000
- 1 ≤ `d` ≤ 128
- `Q`、`K` 和 `V` 中的所有元素从 `[-100.0, 100.0]` 范围内采样
- 所有矩阵的数据类型为 `float32`
- 性能测试使用 `M` = 5,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {

}
```
