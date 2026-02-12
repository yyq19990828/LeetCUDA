#include <cuda_runtime.h>

// 定义向量化访问宏
#define UCHAR4(value) (reinterpret_cast<uchar4*>(&(value))[0])

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (idx >= totalPixels) return;

    // 使用宏进行向量化读取，一次处理一个完整像素 (R,G,B,A)
    uchar4 pixel = UCHAR4(image[idx * 4]);

    pixel.x = 255 - pixel.x;  // R
    pixel.y = 255 - pixel.y;  // G
    pixel.z = 255 - pixel.z;  // B
    // pixel.w (Alpha) 保持不变

    // 使用宏进行向量化写入
    UCHAR4(image[idx * 4]) = pixel;
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int totalPixels = width * height;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
