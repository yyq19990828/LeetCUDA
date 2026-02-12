# K-Means Clustering

> LeetGPU: https://leetgpu.com/challenges/k-means-clustering

## 难度

Hard

## 题目描述

实现二维点的 K-Means 聚类算法。给定数据点的 x 和 y 坐标数组、初始质心以及其他参数，将每个点分配到最近的质心，并迭代更新质心。最终的质心和标签应存储在输出数组中。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `labels`、`final_centroid_x` 和 `final_centroid_y` 中

## 示例

**示例 1：**

```
Input:
sample_size = 4, k = 2, max_iterations = 10
data_x = [1.0, 2.0, 8.0, 9.0]
data_y = [1.0, 2.0, 8.0, 9.0]
initial_centroid_x = [1.0, 8.0]
initial_centroid_y = [1.0, 8.0]
Output: (see reference implementation for expected output)
```

## 约束条件

- 1 ≤ sample_size ≤ 1000000
- 1 ≤ k ≤ 1000
- 所有数组为 float32 类型，labels 除外，为 int32 类型
- 性能测试使用 `k` = 5, `max_iterations` = 30, `sample_size` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {}
```
