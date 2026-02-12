# 1D Convolution

> LeetGPU: https://leetgpu.com/challenges/1d-convolution

## 难度

Easy

## 题目描述

实现一个执行一维卷积运算的程序。给定一个输入数组和一个卷积核（滤波器），计算卷积输出。卷积应使用 "valid" 边界条件，即卷积核仅在与输入完全重叠的位置进行计算。输入包含两个数组：`input`：一个 32 位浮点数的一维数组；`kernel`：一个 32 位浮点数的一维数组，表示卷积核。输出应写入 `output` 数组，其大小为 `input_size - kernel_size + 1`。卷积运算的数学定义为：output[i] = sum(input[i + j] * kernel[j])，其中 j 从 0 到 kernel_size - 1，i 的范围为 0 到 input_size - kernel_size。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在数组 `output` 中

## 示例

**示例 1：**

```
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]
```

**示例 2：**

```
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]
```

## 约束条件

- 1 ≤ `input_size` ≤ 1,500,000
- 1 ≤ `kernel_size` ≤ 2047
- `kernel_size` ≤ `input_size`
- 性能测试使用 `input_size` = 1,500,000，`kernel_size` = 2,047

## 代码模板

```cpp
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
```
