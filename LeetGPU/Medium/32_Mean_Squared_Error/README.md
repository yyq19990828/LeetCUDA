# Mean Squared Error

> LeetGPU: https://leetgpu.com/challenges/mean-squared-error

## 难度

Medium

## 题目描述

实现一个 GPU 程序，计算预测值和目标值之间的均方误差（MSE）。给定两个等长数组 `predictions` 和 `targets`，计算：MSE = (1/N) * sum((predictions_i - targets_i)^2, i=1..N)，其中 N 是每个数组的元素数量。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `mse` 变量中

## 示例

**示例 1：**

```
Input:  predictions = [1.0, 2.0, 3.0, 4.0]
          targets = [1.5, 2.5, 3.5, 4.5]
  Output: mse = 0.25
```

**示例 2：**

```
Input:  predictions = [10.0, 20.0, 30.0]
          targets = [12.0, 18.0, 33.0]
  Output: mse = 5.67
```

## 约束条件

- 1 ≤ `N` ≤ 100,000,000
- -1000.0 ≤ `predictions[i]`, `targets[i]` ≤ 1000.0
- 性能测试使用 `N` = 50,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {}
```
