// op3.cu -- Optimization: FP16 arithmetic (4 points)
//
// The input and the mask are converted to half precision on the GPU
// (float2halfArray), and the convolution's multiply-adds are done with FP16
// intrinsics (__hfma). Half precision has only an 11-bit mantissa, so a long
// FP16 running sum loses accuracy quickly. Each thread therefore accumulates
// the K*K products of one input channel in half, then adds that partial sum
// to a float accumulator. The FP16 math does most of the work while the
// result stays close to the FP32 reference (relative error ~1e-3). The output
// is written directly as float, so no half->float pass over the output is
// needed.
//
// Bugs fixed vs. the previous version:
//   * tile index taken from blockIdx.z (the batch) and batch/feature map read
//     from blockIdx.x/blockIdx.y, inconsistent with the dimGrid(B, M, tiles) launch
//   * `sum += __hadd(__hmul(x, w), sum)` added the running sum to itself (sum*2 + x*w)
//   * input index ignored the stride S (p+h instead of h*S+p)
//   * conversion kernels used a fixed 16 blocks -> very slow for 5000 images
#include <cmath>
#include <iostream>
#include <algorithm>
#include "cuda_fp16.h"
#include "gpu-new-forward.h"

#define BLOCK_SIZE 16
#define CONVERT_THREADS 256

__global__ void conv_forward_kernel(float *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = (blockIdx.z / W_grid) * BLOCK_SIZE + threadIdx.y;
    const int w = (blockIdx.z % W_grid) * BLOCK_SIZE + threadIdx.x;

    if (h < H_out && w < W_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            half partial = __float2half(0.0f);
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    partial = __hfma(in_4d(b, c, h * S + p, w * S + q), mask_4d(m, c, p, q), partial);
            acc += __half2float(partial);
        }
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

// grid-stride conversion loop
__global__ void float2halfArray(const float *input, half *output, const long long size)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < size; i += (long long)blockDim.x * gridDim.x)
        output[i] = __float2half(input[i]);
}

static int convert_blocks(long long n)
{
    return (int)std::min((n + CONVERT_THREADS - 1) / CONVERT_THREADS, 65535LL * 16);
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
    const long long in_n = (long long)B * C * H * W;
    const long long mask_n = (long long)M * C * K * K;

    half *device_input_half;
    half *device_mask_half;
    cudaMalloc((void **)&device_input_half, in_n * sizeof(half));
    cudaMalloc((void **)&device_mask_half, mask_n * sizeof(half));
    check_cuda("malloc half buffers");

    float2halfArray<<<convert_blocks(in_n), CONVERT_THREADS>>>(device_input, device_input_half, in_n);
    float2halfArray<<<convert_blocks(mask_n), CONVERT_THREADS>>>(device_mask, device_mask_half, mask_n);

    const int tiles = ((H_out + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((W_out + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimGrid(B, M, tiles);   // B can be 10000: fine for grid.x (limit 2^31-1)
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input_half, device_mask_half, B, M, C, H, W, K, S);
    check_cuda("fp16 conv launch");

    cudaFree(device_input_half);   // cudaFree synchronizes implicitly
    cudaFree(device_mask_half);
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
