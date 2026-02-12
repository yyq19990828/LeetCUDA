# Monte Carlo Integration

> LeetGPU: https://leetgpu.com/challenges/monte-carlo-integration

## 难度

Medium

## 题目描述

在 GPU 上实现蒙特卡洛积分。给定一组函数值 y_i = f(x_i)，其中 x_i 为在区间 [a, b] 上均匀分布的随机采样点，估计定积分：integral(a, b) f(x) dx ≈ (b - a) * (1/n) * sum(y_i)。蒙特卡洛方法通过计算函数值的平均值并乘以区间宽度来近似积分。

## 实现要求

- 不允许使用外部库
- `solve` 函数签名不可修改
- 最终结果必须存储在 `result` 变量中
- 使用绝对容差 1e-2 和相对容差 1e-2 进行测试

## 示例

**示例：**

```
Input:  a = 0, b = 2, n_samples = 8
        y_samples = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]
Output: result = 3.1875
```

## 约束条件

- 1 ≤ `n_samples` ≤ 100,000,000
- -1000.0 ≤ `a` < `b` ≤ 1000.0
- -10000.0 ≤ 函数值 ≤ 10000.0
- 容差设置为 1e-2，以适应蒙特卡洛方法固有的随机性和浮点精度偏差。
- 性能测试使用 `n_samples` = 10,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {}
```
