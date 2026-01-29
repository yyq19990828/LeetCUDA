---
name: cuda-kernel
description: Create new CUDA kernel templates with PyTorch bindings and performance tests
user-invocable: true
---

# CUDA Kernel Creation Wizard

Create new CUDA kernels following the LeetCUDA project's code style and best practices.

## What Gets Created

1. **`.cu` file** - CUDA kernel implementation
2. **`.py` file** - PyTorch test and benchmark script

## Code Style Requirements

### CUDA File (.cu)

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <torch/types.h>

// Standard memory access macros
#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
```

### Python Test File (.py)

```python
import torch
import time
from torch.utils.cpp_extension import load

# Compile CUDA extension
lib = load(
    name='kernel_name_lib',
    sources=['kernel_name.cu'],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math"
    ],
    extra_cflags=['-std=c++17', '-O3']
)

# Performance benchmark
def benchmark():
    # Compare with PyTorch native implementation
    pass
```

## Directory Structure

New kernels should be placed in `kernels/<kernel-name>/`:

```
kernels/
└── <kernel-name>/
    ├── <kernel-name>.cu    # CUDA implementation
    ├── <kernel-name>.py    # Python test
    └── README.md           # Optional: algorithm explanation
```

## Optimization Checklist

Consider these optimizations when creating kernels:

- [ ] Use shared memory to reduce global memory access
- [ ] Coalesced memory access patterns
- [ ] Avoid bank conflicts
- [ ] Use vectorized loads (float4, int4)
- [ ] Loop unrolling (#pragma unroll)
- [ ] Avoid warp divergence
- [ ] Proper block and grid sizing
