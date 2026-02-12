# Attention with Linear Biases

> LeetGPU: https://leetgpu.com/challenges/attention-with-linear-biases

## 难度

Medium

## 题目描述

实现 Attention with Linear Biases (ALiBi)，遵循论文"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"中描述的方法。给定大小为 `M×d` 的查询矩阵 `Q`、大小为 `N×d` 的键矩阵 `K` 和大小为 `N×d` 的值矩阵 `V`，你的程序应使用以下公式计算输出矩阵：

Attention_ALiBi(Q, K, V) = softmax(QK^T / sqrt(d) + alpha * Delta) * V

其中 alpha 是控制线性偏置的斜率，`Delta = i - j` 表示查询位置 `i` 和键位置 `j` 之间的相对位置。softmax 函数按行应用。`Q`、`K`、`V`、`output` 和 `alpha` 的数据类型均为 `float32`；`M`、`N`、`d` 的数据类型为 `int32`。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在输出矩阵 `output` 中

## 示例

**示例 1：**

```
Input: `Q` (2×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \end{bmatrix} \] `K` (3×4): \[ \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 1.0 & 0.0 \end{bmatrix} \] `V` (3×4): \[ \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 \\ 5.0 & 6.0 & 7.0 & 8.0 \\ 9.0 & 10.0 & 11.0 & 12.0 \end{bmatrix} \] \(\alpha = 0.5\) Output: `output` (2×4): \[ \begin{bmatrix} 3.05 & 4.05 & 6.05 & 7.05 \\ 3.93 & 4.93 & 5.93 & 6.93 \end{bmatrix} \]
```

**示例 2：**

```
Input: `Q` (1×2): \[ \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \] `K` (2×2): \[ \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \] `V` (2×2): \[ \begin{bmatrix} 3.0 & 4.0 \\ 5.0 & 6.0 \end{bmatrix} \] `α` = 0.8 Output: `output` (1×2): \[ \begin{bmatrix} 3.95 & 4.95 \end{bmatrix} \]
```

## 约束条件

- 矩阵 `Q` 的大小为 `M×d`，矩阵 `K` 和 `V` 的大小为 `N×d`
- 1 ≤ `M`, `N` ≤ 2048
- 1 ≤ `d` ≤ 1024
- -1.0 ≤ `alpha` ≤ 1.0
- 性能测试使用 `M` = 2,048, `N` = 2,048

## 代码模板

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {}
```
