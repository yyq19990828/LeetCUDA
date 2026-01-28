# CUDA Kernel Python 绑定详解

本文档详细分析 `dot_product.cu` 中 PyTorch/pybind11 绑定的实现方式。

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 头文件依赖](#2-头文件依赖)
- [3. 辅助宏定义](#3-辅助宏定义)
- [4. Torch 绑定宏详解](#4-torch-绑定宏详解)
- [5. PYBIND11_MODULE 入口](#5-pybind11_module-入口)
- [6. Python 端调用](#6-python-端调用)
- [7. JIT 编译流程](#7-jit-编译流程)
- [8. 数据类型转换](#8-数据类型转换)

---

## 1. 整体架构

整个绑定系统由三个层次组成：

```
┌─────────────────────────────────────────────────────────┐
│                    Python 层                             │
│   lib.dot_prod_f32_f32(a, b)                            │
└─────────────────────┬───────────────────────────────────┘
                      │ torch.utils.cpp_extension.load()
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  PyTorch C++ 层                          │
│   torch::Tensor dot_prod_f32_f32(Tensor a, Tensor b)    │
│   - 类型检查 (CHECK_TORCH_TENSOR_DTYPE)                  │
│   - 输出 Tensor 创建                                     │
│   - Grid/Block 维度计算                                  │
│   - Kernel 启动                                          │
└─────────────────────┬───────────────────────────────────┘
                      │ <<<grid, block>>>
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   CUDA Kernel 层                         │
│   dot_prod_f32_f32_kernel<NUM_THREADS>                  │
│   - Warp Reduce                                         │
│   - Shared Memory Reduce                                │
│   - atomicAdd 汇总                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 头文件依赖

```cpp
#include <torch/extension.h>
#include <torch/types.h>
```

### torch/extension.h

这是 PyTorch C++ 扩展的核心头文件，包含：

- **pybind11**: Python/C++ 绑定库
- **torch::Tensor**: PyTorch 张量的 C++ 表示
- **TORCH_EXTENSION_NAME**: 编译时自动定义的模块名宏

### torch/types.h

提供 PyTorch 数据类型定义：

- `torch::kFloat32` - 32位浮点
- `torch::kHalf` - 16位浮点 (FP16)
- `torch::kCUDA` - CUDA 设备类型

---

## 3. 辅助宏定义

### 3.1 STRINGFY 宏

```cpp
#define STRINGFY(str) #str
```

**作用**: 将宏参数转换为字符串字面量。

**示例**:
```cpp
STRINGFY(dot_prod_f32_f32)  // 展开为 "dot_prod_f32_f32"
```

### 3.2 TORCH_BINDING_COMMON_EXTENSION 宏

```cpp
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));
```

**作用**: 向 Python 模块注册一个 C++ 函数。

**参数说明**:
- `m`: pybind11 模块对象
- `STRINGFY(func)`: Python 中的函数名（字符串）
- `&func`: C++ 函数指针
- `STRINGFY(func)`: 文档字符串（可选）

**展开示例**:
```cpp
TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
// 展开为:
m.def("dot_prod_f32_f32", &dot_prod_f32_f32, "dot_prod_f32_f32");
```

### 3.3 CHECK_TORCH_TENSOR_DTYPE 宏

```cpp
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }
```

**作用**: 运行时检查张量数据类型是否正确。

**使用示例**:
```cpp
CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
// 如果 a 不是 float32 类型，抛出异常
```

---

## 4. Torch 绑定宏详解

### 4.1 TORCH_BINDING_DOT_PROD 宏

这是最核心的宏，用于生成完整的绑定函数：

```cpp
#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type, \
                               n_elements)                                    \
  torch::Tensor dot_prod_##packed_type##_##acc_type(torch::Tensor a,          \
                                                    torch::Tensor b) {        \
    /* 实现 */                                                                 \
  }
```

**参数说明**:

| 参数 | 含义 | 示例 |
|------|------|------|
| `packed_type` | 打包类型标识 | `f32`, `f32x4`, `f16x8_pack` |
| `acc_type` | 累加器类型标识 | `f32` |
| `th_type` | PyTorch 数据类型 | `torch::kFloat32`, `torch::kHalf` |
| `element_type` | CUDA 元素类型 | `float`, `half` |
| `n_elements` | 每个线程处理的元素数 | `1`, `4`, `8` |

### 4.2 函数体详解

```cpp
torch::Tensor dot_prod_f32_f32(torch::Tensor a, torch::Tensor b) {
    // 1. 类型检查
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)

    // 2. 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)   // 数据类型
        .device(torch::kCUDA, 0); // CUDA 设备 0
    auto prod = torch::zeros({1}, options);  // 标量结果

    // 3. 获取维度信息
    const int ndim = a.dim();

    // 4. 根据维度选择执行路径
    if (ndim != 2) {
        // 一维或多维展平处理
        int N = 1;
        for (int i = 0; i < ndim; ++i) {
            N *= a.size(i);  // 计算总元素数
        }
        dim3 block(256);
        dim3 grid(((N + 256 - 1) / 256) / n_elements);

        // 启动 kernel
        dot_prod_f32_f32_kernel<256><<<grid, block>>>(
            reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            prod.data_ptr<float>(),
            N
        );
    } else {
        // 二维张量优化路径
        const int S = a.size(0);  // 行数
        const int K = a.size(1);  // 列数
        const int N = S * K;

        if ((K / n_elements) <= 1024) {
            // 使用 DISPATCH 宏选择合适的线程数
            DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type,
                                     element_type, n_elements)
        } else {
            // 回退到默认配置
            // ...
        }
    }

    return prod;
}
```

### 4.3 DISPATCH_DOT_PROD_KERNEL 宏

根据输入大小动态选择线程块大小：

```cpp
#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type, \
                                 n_elements)                              \
  const int NT = (K) / (n_elements);                                      \
  dim3 block(NT);                                                         \
  dim3 grid((S));                                                         \
  switch (NT) {                                                           \
  case 32:                                                                \
    LANUCH_DOT_PROD_KERNEL(32, packed_type, acc_type, element_type)       \
    break;                                                                \
  case 64:                                                                \
    LANUCH_DOT_PROD_KERNEL(64, packed_type, acc_type, element_type)       \
    break;                                                                \
  /* ... 128, 256, 512, 1024 ... */                                       \
  }
```

**为什么需要 switch-case？**

CUDA kernel 的模板参数 `NUM_THREADS` 必须是编译时常量。通过 switch-case，每个 case 分支都实例化一个特定线程数的 kernel 版本。

### 4.4 LANUCH_DOT_PROD_KERNEL 宏

实际启动 CUDA kernel：

```cpp
#define LANUCH_DOT_PROD_KERNEL(NT, packed_type, acc_type, element_type)    \
  dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                       \
      <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),    \
                        reinterpret_cast<element_type *>(b.data_ptr()),    \
                        prod.data_ptr<float>(), N);
```

**关键点**:
- `<<<grid, block>>>`: CUDA kernel 启动语法
- `reinterpret_cast`: 将 `void*` 转换为具体类型指针
- `a.data_ptr()`: 获取张量底层数据指针

---

## 5. PYBIND11_MODULE 入口

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x8_pack_f32)
}
```

**说明**:

- `PYBIND11_MODULE`: pybind11 宏，定义 Python 模块入口
- `TORCH_EXTENSION_NAME`: 由编译系统自动定义，对应 `load()` 中的 `name` 参数
- `m`: pybind11 模块对象，用于注册函数

---

## 6. Python 端调用

### 6.1 JIT 编译加载

```python
from torch.utils.cpp_extension import load

lib = load(
    name="dot_product_lib",           # 模块名
    sources=["dot_product.cu"],       # 源文件
    extra_cuda_cflags=[               # NVCC 编译选项
        "-O3",                        # 优化级别
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",   # 允许 constexpr 扩展
        "--expt-extended-lambda",     # 允许扩展 lambda
        "--use_fast_math",            # 快速数学运算
    ],
    extra_cflags=["-std=c++17"],      # C++ 编译选项
)
```

### 6.2 调用示例

```python
import torch

# 创建输入张量
a = torch.randn((1024, 1024)).cuda().float()
b = torch.randn((1024, 1024)).cuda().float()

# 调用自定义 kernel
result = lib.dot_prod_f32_f32(a, b)

# 对比 PyTorch 原生实现
expected = torch.dot(a.flatten(), b.flatten())
```

---

## 7. JIT 编译流程

```
┌─────────────────────────────────────────────────────────────┐
│  torch.utils.cpp_extension.load()                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 检查缓存 (~/.cache/torch_extensions/)                    │
│     - 如果已编译且源文件未修改，直接加载                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ (缓存未命中)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 生成 setup.py 和构建脚本                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. NVCC 编译 CUDA 代码                                      │
│     nvcc -O3 --use_fast_math ... -c dot_product.cu          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 链接生成 .so 共享库                                      │
│     g++ -shared -o dot_product_lib.so ...                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Python import 加载模块                                   │
│     import dot_product_lib                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 数据类型转换

### 8.1 PyTorch Tensor → CUDA 指针

```cpp
// torch::Tensor → 原始指针
float* ptr = reinterpret_cast<float*>(a.data_ptr());

// 或使用模板方法
float* ptr = a.data_ptr<float>();
```

### 8.2 类型映射表

| PyTorch 类型 | CUDA 类型 | 绑定参数 |
|-------------|----------|----------|
| `torch::kFloat32` | `float` | `element_type=float` |
| `torch::kHalf` | `half` / `__half` | `element_type=half` |
| `torch::kBFloat16` | `__nv_bfloat16` | - |
| `torch::kFloat64` | `double` | - |

### 8.3 向量化类型

```cpp
// 128-bit 加载/存储
float4 reg = FLOAT4(data[idx]);    // 4 x float32 = 128 bits
half2 reg = HALF2(data[idx]);      // 2 x float16 = 32 bits
LDST128BITS(pack[0]) = LDST128BITS(data[idx]);  // 128 bits
```

---

## 总结

Python 绑定的关键组件：

1. **头文件**: `torch/extension.h` 提供 pybind11 + PyTorch 集成
2. **类型检查**: `CHECK_TORCH_TENSOR_DTYPE` 确保输入类型正确
3. **函数生成**: `TORCH_BINDING_DOT_PROD` 宏批量生成绑定函数
4. **Kernel 分发**: `DISPATCH_DOT_PROD_KERNEL` 根据大小选择最优配置
5. **模块注册**: `PYBIND11_MODULE` 导出函数到 Python
6. **JIT 编译**: `torch.utils.cpp_extension.load()` 动态编译加载

这种设计模式在 CUDA 性能优化库中非常常见，它将：
- **底层优化** (CUDA kernel)
- **类型安全** (PyTorch 张量检查)
- **易用性** (Python 接口)

三者有机结合，是 PyTorch 自定义算子的标准实践。
