# Sorting

> LeetGPU: https://leetgpu.com/challenges/sorting

## 难度

Medium

## 题目描述

编写一个程序，将 32 位浮点数数组按升序排序。你可以自由选择任何排序算法。

## 实现要求

- 只允许使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 排序结果必须存储回输入的 `data` 数组中

## 示例

**示例：**

```
Input: data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0], N = 6
Output: data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]
```

## 约束条件

- 1 ≤ `N` ≤ 1,000,000
- 性能测试使用 `N` = 1,000,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// data is device pointer
extern "C" void solve(float* data, int N) {}
```
