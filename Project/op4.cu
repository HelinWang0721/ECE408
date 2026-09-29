// op4.cu -- Optimization: Input channel reduction: tree (3 points)
//
// The C input channels are split across the threadIdx.z dimension of the
// block: thread (x, y, z) computes the partial sum of output pixel (h, w) over
// channels c = z, z + CZ, z + 2*CZ, ... (CZ = blockDim.z, a power of two).
// The CZ partial sums of a pixel are stored in shared memory and combined
// with a tree reduction in log2(CZ) steps; thread z == 0 writes the result.
// This exposes C times more parallelism per output pixel, which helps when
// H_out * W_out is small and C is large.
//
// Block: (T, T, CZ) with T*T*CZ <= 1024, where CZ = next_pow2(C) (<= 64).
// Dynamic shared memory: T*T*CZ floats.
//
// Bugs fixed vs. the previous version: it launched blockDim.z == 1 and 0 bytes
// of dynamic shared memory, stored partial sums inside the channel loop,
// indexed shared memory with different argument orders on write and read,
// wrote out_4d(blockIdx.x, blockIdx.y, ...) (map/tile instead of batch/map),
// and called __syncthreads() inside divergent branches.
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int T = blockDim.x;      // output tile width (blockDim.x == blockDim.y)
    const int CZ = blockDim.z;     // channel lanes, power of two

    extern __shared__ float partial[];   // [CZ][T][T]

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tree(z, y, x) partial[((z) * T + (y)) * T + (x)]

    const int W_grid = (W_out + T - 1) / T;
    const int m = blockIdx.x;
    const int b = blockIdx.z;
    const int h = (blockIdx.y / W_grid) * T + threadIdx.y;
    const int w = (blockIdx.y % W_grid) * T + threadIdx.x;
    const bool valid = (h < H_out && w < W_out);

    float sum = 0.0f;
    if (valid) {
        for (int c = threadIdx.z; c < C; c += CZ)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    sum += in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
    }
    tree(threadIdx.z, threadIdx.y, threadIdx.x) = sum;

    // tree reduction over z; every thread reaches every barrier
    for (int stride = CZ / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.z < stride)
            tree(threadIdx.z, threadIdx.y, threadIdx.x) += tree(threadIdx.z + stride, threadIdx.y, threadIdx.x);
    }

    if (threadIdx.z == 0 && valid)
        out_4d(b, m, h, w) = tree(0, threadIdx.y, threadIdx.x);   // CZ==1: own value, no barrier needed

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tree
}


static void check_cuda(const char *where)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (" << where << "): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    cudaMalloc((void **)device_output_ptr, (size_t)B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **)device_input_ptr, (size_t)B * C * H * W * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, (size_t)M * C * K * K * sizeof(float));
    check_cuda("prolog malloc");

    cudaMemcpy(*device_input_ptr, host_input, (size_t)B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (size_t)M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda("prolog memcpy");
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    int CZ = 1;
    while (CZ < C && CZ < 64) CZ *= 2;           // channel lanes (power of two)
    int T = 16;
    while (T > 1 && T * T * CZ > 1024) T /= 2;   // keep <= 1024 threads per block

    const int W_grid = (W_out + T - 1) / T;
    const int H_grid = (H_out + T - 1) / T;
    dim3 blockDim(T, T, CZ);
    dim3 gridDim(M, H_grid * W_grid, B);
    const size_t smem = (size_t)T * T * CZ * sizeof(float);
    conv_forward_kernel<<<gridDim, blockDim, smem>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    check_cuda("tree-reduction conv launch");
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    cudaMemcpy(host_output, device_output, (size_t)B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
