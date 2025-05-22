## ⚡️⚡️Toy-HGEMM: 实现 cuBLAS 98%~100% 的 TFLOPS 🎉🎉

![toy-hgemm-library](https://github.com/user-attachments/assets/962bda14-b494-4423-b8eb-775da9f5503d)

[📖Toy-HGEMM Library⚡️⚡️](./kernels/hgemm) 是一个从头开始使用 Tensor Cores (WMMA, MMA PTX 和 CuTe API) 编写了许多 HGEMM 内核的库，因此可以达到 **cuBLAS** `98%~100%` 的性能。这里的代码源自 📖[CUDA-Learn-Notes](https://github.com/xlite-dev/CUDA-Learn-Notes) ![](https://img.shields.io/github/stars/xlite-dev/CUDA-Learn-Notes.svg?style=social) 并作为一个独立的库导出，请查看 [CUDA-Learn-Notes](https://github.com/xlite-dev/CUDA-Learn-Notes) 获取最新更新。欢迎 🌟👆🏻star 这个仓库来支持我，非常感谢 ~ 🎉🎉

<div id="hgemm-sgemm"></div>

<div align='center'>
  <img src='https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078' height="170px" width="270px">
</div>


目前，在 NVIDIA L20, RTX 4090 和 RTX 3080 Laptop 上，与 cuBLAS 默认的 Tensor Cores 数学算法 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` 相比，本仓库实现的 `HGEMM (WMMA/MMA/CuTe)` (`蓝色`🔵) 可以达到其 (`橙色`🟠) 性能的 `98%~100%`。更多详情请查看 [toy-hgemm library⚡️⚡️](./kernels/hgemm)。

|📚特性 |📚特性 |📚特性 |📚特性|
|:---:|:---:|:---:|:---:|
|✔️CUDA/**Tensor Cores**|✔️K 维度循环|✔️分块 Block(BMxBK)|✔️分块 Threads(T 8x8)|
|✔️WMMA(m16n16k16)|✔️MMA(m16n8k16)|✔️Pack LDST(128 bits)|✔️SMEM Padding|
|✔️Copy Async|✔️分块 MMAs|✔️分块 Warps|✔️**多阶段(2~4)**|
|✔️寄存器双缓冲|✔️**Block Swizzle**|✔️**Warp Swizzle**|✔️**SMEM Swizzle**(CuTe/MMA)|
|✔️Collective Store(Shfl)|✔️Layout NN|✔️Layout TN|✔️SGEMM FP32/TF32|

## ©️引用🎉🎉

```BibTeX
@misc{hgemm-tensorcores-mma@2024,
  title={hgemm-tensorcores-mma: Write HGEMM from scratch using Tensor Cores with WMMA, MMA PTX and CuTe API.},
  url={https://github.com/xlite-dev/hgemm-tensorcores-mma},
  note={Open-source software available at https://github.com/xlite-dev/hgemm-tensorcores-mma},
  author={xlite-dev etc},
  year={2024}
}
```

## 📖 Toy-HGEMM 库中的 HGEMM CUDA 内核 🎉🎉

<div id="kernels"></div>

```C++
void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle, int swizzle_stride);
```

## 📖 目录

- [📖 前提条件](#prerequisites)
- [📖 安装](#install)
- [📖 Python 测试](#test)
- [📖 C++ 测试](#test-cpp)
- [📖 NVIDIA L20 性能测试](#perf-l20)
- [📖 NVIDIA RTX 4090 性能测试](#perf-4090)
- [📖 NVIDIA RTX 3080 Laptop 性能测试](#perf-3080)
- [📖 性能优化笔记](#opt-docs)
- [📖 参考资料](#ref)

## 📖 前提条件
<div id="prerequisites"></div>

- PyTorch >= 2.0, CUDA >= 12.0
- 推荐: PyTorch 2.5.1, CUDA 12.5

## 📖 安装

<div id="install"></div>

本仓库实现的 HGEMM 可以作为一个 Python 库安装，即 `toy-hgemm` 库 (可选)。
```bash
cd kernels/hgemm
git submodule update --init --recursive --force # Fetch `CUTLASS` submodule， needed
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl # pip uninstall toy-hgemm -y
```

## 📖 Python 测试

<div id="test"></div>

**CUTLASS**: Fetch `CUTLASS` submodule. 目前，我使用 `v3.5.1` 用于 HGEMM CuTe 内核。
```bash
git submodule update --init --recursive --force
```

您可以通过 Python 脚本测试许多自定义 HGEMM 内核，并找出它们性能上的差异。

```bash
# 您可以只测试 Ada 或 Ampere，也可以测试 Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada # 仅用于 Ada
export TORCH_CUDA_ARCH_LIST=Ampere # 仅用于 Ampere
python3 hgemm.py --wmma # 测试所有 MNK 的默认 wmma 内核
python3 hgemm.py --mma  # 测试所有 MNK 的默认 mma 内核
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --wmma # 测试特定 MNK 的默认 wmma 内核
python3 hgemm.py --M 16384 --N 16384 --K 8192 --i 10 --mma # 测试特定 MNK 的默认 mma 内核
python3 hgemm.py --wmma-all # 测试所有 MNK 的所有 wmma 内核
python3 hgemm.py --mma-all # 测试所有 MNK 的所有 mma 内核
python3 hgemm.py --cuda-all --wmma-all --mma-all # 测试所有 MNK 的所有内核
python3 hgemm.py --cute-tn --no-default # 测试所有 MNK 的带有 smem swizzle 的 cute hgemm 内核
```
如果您想绘制 TFLOPS 曲线，您需要先安装 `matplotlib` 并设置 --plot-flops (或 --plot) 选项。
```bash
python3 -m pip install matplotlib
# 指定 topk 只绘制性能最好的前 k 个内核。
python3 hgemm.py --mma-all --plot --topk 8
# 测试默认 mma 内核和带有 smem swizzle 的 cute hgemm 内核，适用于所有 MNK
python3 hgemm.py --cute-tn --mma --plot
```

## 📖 C++ 测试

<div id="test-cpp"></div>

HGEMM 基准测试也支持 C++ 测试。目前，它支持以下实现之间的比较：

- 本仓库实现的 MMA HGEMM NN
- 本仓库实现的 CuTe HGEMM TN
- 使用默认 Tensor Cores 数学算法的 cuBLAS HGEMM TN

从 C++ 二进制测试获得性能数据往往比 Python 测试略好。这种差异可能归因于 PyTorch Python 绑定引入的额外开销。
```bash
make
./hgemm_mma_stage.bin
# NVIDIA L20
ALGO = MMA16816 HGEMM NN MMA=2x4 WARP=4x4x2 STAGES=2 BLOCK SWIZZLE=2048
M N K =  12544  12544  12544, Time =   0.03445555   0.03446098   0.03447399 s, AVG Performance =   114.5541 Tflops
M N K =  15360  15360  15360, Time =   0.06307226   0.06307789   0.06308864 s, AVG Performance =   114.9017 Tflops
M N K =  15616  15616  15616, Time =   0.06612480   0.06612798   0.06613094 s, AVG Performance =   115.1739 Tflops
M N K =  15872  15872  15872, Time =   0.06969549   0.06970215   0.06971290 s, AVG Performance =   114.7305 Tflops
M N K =  16128  16128  16128, Time =   0.07295078   0.07295406   0.07295693 s, AVG Performance =   115.0064 Tflops
M N K =  16384  16384  16384, Time =   0.07663001   0.07663534   0.07664947 s, AVG Performance =   114.7785 Tflops

./hgemm_cute.bin
# NVIDIA L20
ALGO = CuTe HGEMM, TN, STAGES=2, SMEM SWIZZLE=<3, 3, 3>, BLOCK SWIZZLE=2048
M N K =  12544  12544  12544, Time =   0.03413504   0.03414354   0.03415450 s, AVG Performance =   115.6191 Tflops
M N K =  15360  15360  15360, Time =   0.06227354   0.06228111   0.06228992 s, AVG Performance =   116.3717 Tflops
M N K =  15616  15616  15616, Time =   0.06492467   0.06493727   0.06496666 s, AVG Performance =   117.2858 Tflops
M N K =  15872  15872  15872, Time =   0.06843085   0.06843873   0.06844723 s, AVG Performance =   116.8485 Tflops
M N K =  16128  16128  16128, Time =   0.07200256   0.07200881   0.07201792 s, AVG Performance =   116.5161 Tflops
M N K =  16384  16384  16384, Time =   0.07564493   0.07565752   0.07567462 s, AVG Performance =   116.2620 Tflops

./hgemm_cublas.bin
# NVIDIA L20
ALGO = cuBLAS CUBLAS_GEMM_DEFAULT_TENSOR_OP TN
M N K =  12544  12544  12544, Time =   0.03472691   0.03472968   0.03473408 s, AVG Performance =   113.6678 Tflops
M N K =  15360  15360  15360, Time =   0.06332416   0.06333143   0.06334157 s, AVG Performance =   114.4417 Tflops
M N K =  15616  15616  15616, Time =   0.06649446   0.06650184   0.06651699 s, AVG Performance =   114.5264 Tflops
M N K =  15872  15872  15872, Time =   0.06977024   0.06977659   0.06978355 s, AVG Performance =   114.6081 Tflops
M N K =  16128  16128  16128, Time =   0.07319142   0.07320709   0.07326925 s, AVG Performance =   114.6089 Tflops
M N K =  16384  16384  16384, Time =   0.07668429   0.07669371   0.07670784 s, AVG Performance =   114.6912 Tflops
```

## 📖 性能测试

<div id="perf-l20"></div>

### 📖 NVIDIA L20
<!--
目前最优的实现，在L20上（理论Tensor Cores FP16算力为 119.5 TFLOPS），整体上能达到cuBLAS大概`99~100+%`左右的性能。使用WMMA API能达到cuBLAS大概`95%~98%`左右的性能(105-113 TFLOPS vs 105-115 TFLOPS)，使用MMA API能达到115 TFLOPS，部分 case 会超越 cuBLAS。CuTe 版本的 HGEMM 实现了 Block Swizzle（L2 Cache friendly）和 SMEM Swizzle（bank conflicts free），性能最优，大规模矩阵乘能达到 116-117 TFLOPS，是 cuBLAS 大概`98%~100%+`左右的性能，很多case会超越cuBLAS。目前通过 SMEM Padding 和 SMEM Swizzle 的方式缓解 bank conflicts。对于 NN layout，使用 SMEM Padding 缓解 bank conflicts；对于 TN layout，通过 CUTLASS/CuTe 的 SMEM Swizzle 消除 bank conflicts。
-->
目前最优的实现，在 L20 上（理论 Tensor Cores FP16 算力为 119.5 TFLOPS），整体上能达到 cuBLAS 大概 `99%~100+%` 左右的性能。

- 使用 WMMA API，能达到 cuBLAS 大概 `95%~98%` 左右的性能 (105-113 TFLOPS vs 105-115 TFLOPS)。
- 使用 MMA API，能达到 115 TFLOPS，部分 case 会超越 cuBLAS。
- CuTe 版本的 HGEMM 实现了 Block Swizzle（L2 Cache friendly）和 SMEM Swizzle（bank conflicts free），性能最优。大规模矩阵乘能达到 116-117 TFLOPS，是 cuBLAS 大概 `98%~100%+` 左右的性能，很多 case 会超越 cuBLAS。

目前通过 SMEM Padding 和 SMEM Swizzle 的方式缓解 bank conflicts：

- 对于 NN layout，使用 SMEM Padding 缓解 bank conflicts。
- 对于 TN layout，通过 CUTLASS/CuTe 的 SMEM Swizzle 消除 bank conflicts。

<div id="NV-L20"></div>


![NVIDIA_L20_NN+TN+v2](https://github.com/user-attachments/assets/71927ac9-72b3-4ce9-b0e2-788b5885bc99)


测试所有 MNK 设置的命令 (提示: 单独测试每个 MNK 的性能数据更准确。)
```bash
python3 hgemm.py --cute-tn --mma --plot
```

### 📖 NVIDIA GeForce RTX 4090

<div id="perf-4090"></div>

<!--
在NVIDIA RTX 4090上(FP16 Tensor Cores算力为330 TFLOPS)，WMMA(m16n16k16)性能表现比MMA(m16n8k16)要更好，大分部MNK下，本仓库的实现能达到cuBLAS 95%~99%的性能，某些case能超过cuBLAS。就本仓库的实现而言，在RTX 4090上，大规模矩阵乘(MNK>=8192)，WMMA表现更优，小规模矩阵乘，MMA表现更优。
-->

在 NVIDIA RTX 4090 上 (FP16 Tensor Cores 算力为 330 TFLOPS)，WMMA (m16n16k16) 实现的性能表现比 MMA (m16n8k16) 更好。对于大多数 MNK 配置，本仓库的实现能达到 cuBLAS 95%~99% 的性能，某些 case 能超过 cuBLAS。具体来说：

- 对于大规模矩阵乘法 (MNK >= 8192)，WMMA 实现表现更优。
- 对于小规模矩阵乘法，MMA 实现更高效。


![NVIDIA_GeForce_RTX_4090_NN+TN+v4](https://github.com/user-attachments/assets/05ef4f5e-d999-48ea-b58e-782cffb24e85)

```bash
python3 hgemm.py --cute-tn --mma --wmma-all --plot
```

### 📖 NVIDIA GeForce RTX 3080 Laptop

<div id="perf-3080"></div>

<!--
在NVIDIA GeForce RTX 3080 Laptop上测试，使用mma4x4_warp4x4（16 WMMA m16n16k16 ops, warp tile 64x64）以及Thread block swizzle，大部分case能持平甚至超过cuBLAS，使用Windows WSL2 + RTX 3080 Laptop进行测试。
-->
在 NVIDIA GeForce RTX 3080 Laptop 上进行了测试，使用 mma4x4_warp4x4 配置（包括 16 个 WMMA m16n16k16 操作，warp tile 大小为 64x64）以及 Thread block swizzle。在大多数情况下，此设置可以达到甚至超过 cuBLAS 的性能。测试是使用 Windows WSL2 + RTX 3080 Laptop 进行的。

![image](https://github.com/user-attachments/assets/9472e970-c083-4b31-9252-3eeecc761078)

```bash
python3 hgemm.py --wmma-all --plot
```

<details>
<summary> 🔑️ 性能优化笔记(TODO)</summary>

## 📖 性能优化笔记

<div id="opt-docs"></div>

### PyTorch HGEMM Profile

在 Ada 架构下，PyTorch 2.4 使用 matmul 时，会调用:
```C++
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
```
内部实际使用 HMMA(Tensor Cores) 进行计算，在 3080 上 profile 发现使用:
```C++
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize96x64x32_stage3_warpsize2x2x1_tensor16x8x16_kernel
```
因此，只有实现使用 Tensor Cores 的 HGEMM，才有可能接近 PyTorch/cuBLAS 的性能。
```bash
ncu -o hgemm.prof -f python3 bench/prof.py
nsys profile --stats=true -t cuda,osrt,nvtx -o hgemm.prof --force-overwrite true python3 prof.py
```
- SASS (L20)

```C
// ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn_kernel
310	00007f41 37d5b850	      LDSM.16.M88.4 R192, [R169+UR8+0x2000]
311	00007f41 37d5b860	      LDSM.16.M88.4 R196, [R169+UR8+0x2800]
336	00007f41 37d5b9f0	      HMMA.1688.F32 R112, R182, R196, R112
...
```

### SMEM Padding

#### Bank Conflicts 的产生

含义：在访问 shared memory 时，因多个线程读写同一个 Bank 中的不同数据地址时，导致 shared memory 并发读写 退化 成顺序读写的现象叫做 Bank Conflict；

![](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/images/ef322be7c3e5b6b9be69d2b90e88083f50569a58a97129f348e483b946ab4edf.png)

SM 调度单位为一个 warp（一个 warp 内 32 个 Thread），shared_memory 可以 被一个 warp 中的所有（32 个）线程进行访问，shared_memory 映射到大小相等的 32 个 Bank 上，Bank 的数据读取带宽为 32bit / cycle (4 bytes)，因此，主要需要考虑一个 Warp 内 32 线程的访问共享内存时的 bank 冲突。
对于多个线程读取同一个 Bank 数据时（不同地址），硬件把内存读写请求，拆分成 conflict-free requests，进行顺序读写，此时将会触发多次内存事务。特别地，当一个 warp 中的所有线程读写同一个地址时，会触发 broadcast 机制，此时不会退化成顺序读写。上面提到触发 broadcast 机制的条件是 all threads acess same address，但在翻阅 cuda-c-programming-guide 以及最新版本的[NVProfGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 时，发现只要是多个 thread 读写就会触发 broadcast（不需要 All）。

- 多个线程读同一个数据时，仅有一个线程读，然后 broadcast 到其他线程
- 多个线程写同一个数据时，仅会有一个线程写成功

NVIDIA 的[文章](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)中指出，我们还可以通过 `cudaDeviceSetSharedMemConfig()` 函数设置默认 Bank Size（默认为 4 bytes）来避免 bank conflicts，可设置为 cudaSharedMemBankSizeFourByte 或者 cudaSharedMemBankSizeEightByte。对于某些场景来说，设置 cudaSharedMemBankSizeEightByte 或许更加合适，比如使用 double 数据类型时。

```C
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```
目前通过 SMEM Padding 和 SMEM swizzle 的方式缓解 bank conflicts。对于 NN layout，使用 SMEM Padding 缓解 bank conflicts；对于 TN layout，通过 cutlass cute 的 SMEM Swizzle 消除 bank conflicts。

### 双缓冲 Double Buffers

本仓库实现的 HGEMM Double Buffers 策略如下：1）主循环从 bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是 pipeline 的特点决定的；2）由于计算和下一次访存使用的 Shared Memory 不同，因此主循环中每次循环只需要一次 __syncthreads() 即可，对比非 double buffers 版本，总共节省了 ((K + BK - 1) / BK) - 1 次 block 内的同步操作。比如，bk=1 时，HFMA 计算使用的是 s_a[0] 和 s_b[0]，因此，和 s_a[1] 和 s_b[1] 的加载是没有依赖关系的。HFMA 计算，从 global 内存到 s_a[1] 和 s_b[1] 和 HFMA 计算可以并行。s_a[1] 和 s_b[1] 用于加载下一块 BK 需要的数据到共享内存；3）由于 GPU 不能向 CPU 那样支持乱序执行，主循环中需要先将下一次循环计算需要的 Gloabal Memory 中的数据 load 到寄存器，然后进行本次计算，之后再将 load 到寄存器中的数据写到 Shared Memory，这样在 LDG 指令向 Global Memory 做 load 时，不会影响后续 HFMA 及其它运算指令的 launch 执行，也就达到了 Double Buffers 的目的，具体代码见[hgemm.cu](./hgemm.cu)。


### Tile Block

TODO

### Tile Thread

TODO

### Pack LDST 128 bits

TODO

### Async Copy

TODO

### Multi Stages

TODO

### Tensor Cores(WMMA/MMA)

TODO

### Tile MMA/Warp

TODO

### Thread Block Swizze

TODO

### Warp Swizzle

TODO

### Reg Double Buffers

TODO

### Collective Store(Reg Reuse&Warp Shuffle)

TODO

### SMEM Swizzle/Permuted

TODO

</details>

## 📖 参考资料

<div id="ref"></div>

- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
- [cute-gemm](https://github.com/reed-lau/cute-gemm)
- [cutlass_flash_atten_fp8](https://github.com/weishengying/cutlass_flash_atten_fp8)
- [cuda_learning](https://github.com/ifromeast/cuda_learning)
- [cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm)
- [cuda-tensorcore-hgemm](https://github.com/nicolaswilde/cuda-tensorcore-hgemm)
- [How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/sgemv)
- [cute_gemm](https://github.com/weishengying/cute_gemm)
- [cutlass](https://github.com/NVIDIA/cutlass)