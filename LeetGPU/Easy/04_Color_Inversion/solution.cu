#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (idx >= totalPixels) return;

    // 使用 uchar4 向量化读写，一次处理一个完整像素 (R,G,B,A)
    uchar4* pixels = reinterpret_cast<uchar4*>(image);
    uchar4 pixel = pixels[idx];

    pixel.x = 255 - pixel.x;  // R
    pixel.y = 255 - pixel.y;  // G
    pixel.z = 255 - pixel.z;  // B
    // pixel.w (Alpha) 保持不变

    pixels[idx] = pixel;
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int totalPixels = width * height;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
