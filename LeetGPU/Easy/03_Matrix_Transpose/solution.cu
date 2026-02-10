#include <cuda_runtime.h>

#define TILE_SIZE 32

// 使用 shared memory 实现高效转置
// 关键：读写全局内存都是合并访问，通过 shared memory 做中转
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // +1 padding 避免 bank conflict
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // 输入矩阵中的坐标
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;  // col in input
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in input

    // 合并读取：同一 warp 的线程读取连续的列地址
    // 每个线程处理 TILE_SIZE/blockDim.y = 4 行
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if ((y + i) < rows && x < cols) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * cols + x];
        }
    }

    __syncthreads();

    // 输出矩阵中的坐标：block 位置转置
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // col in output
    y = blockIdx.x * TILE_SIZE + threadIdx.y;  // row in output

    // 合并写入：从 tile 中转置读取，写到 output 的连续地址
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if ((y + i) < cols && x < rows) {
            output[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_SIZE, 8);
    dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
                       (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
