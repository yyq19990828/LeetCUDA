# Multi-Agent Simulation

> LeetGPU: https://leetgpu.com/challenges/multi-agent-simulation

## 难度

Hard

## 题目描述

实现一个多智能体群集仿真（boids）程序。输入包括：一个 `agents` 数组，包含 `N` 个智能体，其中 `N` 是智能体总数。每个智能体在数组中占用 4 个连续的 32 位浮点数：[x, y, v_x, v_y]，其中：(x, y) 表示智能体在二维空间中的位置，(v_x, v_y) 表示智能体的速度向量。数组总大小为 `4 * N` 个浮点数，智能体 i 的数据存储在索引 [4i, 4i+1, 4i+2, 4i+3] 处。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 最终结果必须存储在 `agents_next` 数组中

## 示例

**示例 1：**

```
Input: N = 2
agents = [
  0.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 4.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]

Output:
agents_next = [
  1.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 3.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]
```

## 约束条件

- 1 ≤ `N` ≤ 100,000
- 每个智能体的位置和速度分量为 32 位浮点数
- 性能测试使用 `N` = 10,000

## 代码模板

```cpp
#include <cuda_runtime.h>

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {}
```
