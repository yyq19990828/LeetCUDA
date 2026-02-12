# Fast Fourier Transform

> LeetGPU: https://leetgpu.com/challenges/fast-fourier-transform

## 难度

Hard

## 题目描述

编写一个 GPU 程序，计算复数值一维信号的快速傅里叶变换（FFT）。给定一个包含 `N` 个复数的输入 `signal` 数组（以实部/虚部交错存储），计算离散傅里叶变换并将结果存储在 `spectrum` 数组中。FFT 使用以下公式将时域信号转换为频域表示：

X_k = sum(x_n * e^(-j * 2 * pi * k * n / N), n=0..N-1)，其中 k = 0, 1, ..., N-1

FFT 算法通过利用旋转因子的对称性，将计算复杂度从 O(N^2) 降低到 O(N log N)。

## 实现要求

- 不允许使用外部库（包括 cuFFT 等）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `spectrum` 数组中
- 内核必须完全在 GPU 上运行——不允许主机端 FFT 调用
- 输入和输出均使用实部/虚部交错布局：`[real_0, imag_0, real_1, imag_1, ...]`

## 示例

**示例 1：**

```
Input:  N = 4
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        (represents: [1+0j, 0+0j, 0+0j, 0+0j])

Output: spectrum = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j, 1+0j, 1+0j])
```

**示例 2：**

```
Input:  N = 2
        signal = [1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j])

Output: spectrum = [2.0, 0.0, 0.0, 0.0]
        (represents: [2+0j, 0+0j])
```

## 约束条件

- `1 ≤ N ≤ 262,144`
- 所有值为 32 位浮点数
- 绝对误差 ≤ 1e-3 且相对误差 ≤ 1e-3
- 输入和输出数组长度为 `2 × N`
- 性能测试使用 `N` = 262,144

## 代码模板

```cpp
#include <cuda_runtime.h>

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {}
```
