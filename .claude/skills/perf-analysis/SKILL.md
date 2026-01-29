---
name: perf-analysis
description: Analyze CUDA kernel performance and provide optimization suggestions
user-invocable: false
---

# CUDA Performance Analysis Expert

Automatically apply this knowledge when users discuss CUDA kernel performance issues or need optimization advice.

## Performance Analysis Dimensions

### 1. Memory Access Patterns

**Global Memory**:
- Check for coalesced access
- Identify misaligned memory accesses
- Suggest vectorized loads (`float4`, `int4`)

**Shared Memory**:
- Check for bank conflicts
- Analyze if shared memory usage limits occupancy
- Suggest padding strategies to avoid bank conflicts

**Registers**:
- Estimate register usage
- Identify register spilling

### 2. Compute Efficiency

**Warp Efficiency**:
- Identify warp divergence
- Check impact of conditional branches

**Instruction-Level Optimization**:
- Suggest fast math functions (`__expf`, `__logf`)
- Identify fuseable operations (FMA)
- Suggest loop unrolling

### 3. Occupancy Analysis

Consider these limiting factors:
- Max threads per SM
- Max blocks per SM
- Shared memory limits
- Register limits

### 4. Performance Metrics

**Theoretical Peak Calculation**:
```
TFLOPS = num_SMs × cores_per_SM × frequency × 2 (FMA)
```

**Memory Bandwidth**:
```
Bandwidth utilization = actual_bytes_transferred / (time × theoretical_bandwidth)
```

**Arithmetic Intensity**:
```
Arithmetic intensity = FLOPs / memory_bytes_accessed
```

## Common Optimization Suggestions

### GEMM Optimization

1. **Tiling**: Use Block Tile + Warp Tile + Thread Tile
2. **Double Buffering**: Prefetch next tile to shared memory
3. **Vectorization**: Use `LDST128BITS` for 128-bit loads
4. **Tensor Cores**: Use WMMA or MMA PTX for FP16/BF16

### Reduction Optimization

1. **Warp Shuffle**: Use `__shfl_xor_sync` for intra-warp reduction
2. **Sequential Addressing**: Avoid bank conflicts
3. **Multi-level Reduction**: Warp → Block → Grid

### Elementwise Optimization

1. **Vectorization**: One thread processes multiple elements
2. **Grid-Stride Loop**: Handle arbitrary input sizes
3. **Kernel Fusion**: Reduce kernel launch overhead

## Nsight Tools Usage

```bash
# Performance profiling
ncu --set full ./your_program

# Key metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./your_program
```
