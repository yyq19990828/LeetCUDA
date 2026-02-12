# BFS Shortest Path

> LeetGPU: https://leetgpu.com/challenges/bfs-shortest-path

## 难度

Medium

## 题目描述

实现一个程序，使用广度优先搜索（BFS）在无权 2D 网格中寻找最短路径。给定一个包含障碍物的网格以及起点和终点位置，返回到达目标所需的最少步数。

## 实现要求

- 只能使用原生特性（不允许使用外部库）
- `solve` 函数签名不可修改
- 返回最短路径长度，如果不存在路径则返回 -1
- 网格中值为 0 的单元格为可通行，值为 1 的单元格为障碍物
- 允许向 4 个方向移动：上、下、左、右

## 示例

**示例 1：**

```
Input:
  grid (4x4) = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
  ]
  start_row = 0, start_col = 0
  end_row = 3, end_col = 3

Output: 6

Explanation: 一条可能的最短路径：
(0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,3) → (3,3)
```

**示例 2：**

```
Input:
  grid (3x3) = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0]
  ]
  start_row = 0, start_col = 0
  end_row = 0, end_col = 2

Output: -1

Explanation: 由于障碍物完全阻断了通路，不存在可行路径。
```

## 约束条件

- 1 ≤ `rows`, `cols` ≤ 1000
- 网格值为 0（可通行）或 1（障碍物）
- 起点和终点保证在网格范围内且位于可通行单元格上（值为 0）
- 起点和终点可能相同（此时返回 0）
- 性能测试使用 `cols` = 500, `rows` = 500

## 代码模板

```cpp
#include <cuda_runtime.h>

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {}
```
