# Color Inversion

> LeetGPU: https://leetgpu.com/challenges/color-inversion

## 难度

Easy

## 题目描述

编写一个 GPU 程序来反转图像的颜色。图像以 RGBA（红、绿、蓝、Alpha）值的一维数组表示，每个分量是一个 8 位无符号整数（`unsigned char`）。

颜色反转：将每个颜色分量（R, G, B）用 255 减去。**Alpha 分量保持不变**。

输入数组 `image` 包含 `width * height * 4` 个元素。前 4 个元素表示左上角像素的 RGBA 值，接下来 4 个元素表示其右侧像素的值，依此类推。

## 实现要求

- 仅可使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须原地存储在 `image` 数组中

## 示例

**示例 1：**

```
Input:  image = [255, 0, 128, 255, 0, 255, 0, 255], width=1, height=2
Output: [0, 255, 127, 255, 255, 0, 255, 255]

解释：pixel1 [255,0,128,A=255] → [0,255,127,255]
      pixel2 [0,255,0,A=255]   → [255,0,255,255]
```

**示例 2：**

```
Input:  image = [10, 20, 30, 255, 100, 150, 200, 255], width=2, height=1
Output: [245, 235, 225, 255, 155, 105, 55, 255]
```

## 约束条件

- 1 ≤ width ≤ 4096
- 1 ≤ height ≤ 4096
- width * height ≤ 8,388,608
- 性能测试使用 height = 5,120, width = 4,096

## 代码模板

```c++
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

## 解题思路

这是一道 in-place 操作题，每个像素独立处理，无数据依赖。

1. **每线程一像素**：线程 idx 对应第 idx 个像素，操作 `image[idx*4 + 0/1/2]` 三个通道
2. **Alpha 不动**：只反转 R, G, B，跳过 A（偏移量 3）
3. **公式**：`image[base + c] = 255 - image[base + c]`，其中 c ∈ {0, 1, 2}

### 优化方向

- **`uchar4` 向量化访存**：将 4 字节作为一个 `uchar4` 整体读写，减少内存事务
- **`uint` 位运算**：将 4 字节当作一个 32-bit int，用 XOR `0x00FFFFFF` 一次反转 RGB（需注意字节序）
