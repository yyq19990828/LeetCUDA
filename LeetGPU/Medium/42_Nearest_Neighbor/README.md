# Nearest Neighbor

> LeetGPU: https://leetgpu.com/challenges/nearest-neighbor

## 难度

Medium

## 题目描述

实现一个 GPU 程序，对于存储在设备上的 `N` 个三维点，将 `indices[i]` 填充为距离 `points[i]` 最近的点的索引 j（j ≠ i）。比较欧氏距离的平方即可——无需计算平方根。

## 实现要求

- `solve` 函数签名不可修改
- 不允许使用外部库
- 最终结果必须存储在 `indices` 数组中

## 示例

**示例 1：**

```
Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
        indices = [-1, -1, -1]
        N       = 3
Output: indices = [1, 0, 1]   # 0 和 1 互为最近邻，2 距离 1 最近
```

## 约束条件

- 1 ≤ `N` ≤ 100,000
- 坐标为 32 位浮点数，范围为 [-1000, 1000]
- 性能测试使用 `N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {}
```
