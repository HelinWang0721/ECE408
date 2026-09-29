// custom/new-forward.cu -- the implementation that RAI builds and times.
//
// Stacked optimizations (each also exists on its own in Project/opN.cu):
//   * Weight matrix in constant memory (0.5 pt)          -- see op5.cu
//   * __restrict__ + loop unrolling (3 pts)              -- see op5.cu
//   * Multiple kernel implementations for different layer sizes (1 pt):
//       template<KT, CT> fixes the kernel size and channel count at compile
//       time, so the LeNet layers get fully unrolled kernels:
//         layer 1: K=7, C=1   -> conv_forward_kernel<7, 1>
//         layer 2: K=7, C=4   -> conv_forward_kernel<7, 4>
//       anything else (e.g. the 4 test cases) uses the K-only or the fully
//       generic instantiation.
// One thread per output pixel; a 16x16 block covers a 16x16 output tile, so
// the threads of a warp read consecutive input addresses (coalesced for S=1)
// and all read the same mask value (constant-cache broadcast).
//
// Bug fixed vs. the previous version: *device_mask_ptr was never set, so the
// epilog called cudaFree() on an uninitialised pointer. The output store is
// now also outside the channel loop (it was rewritten C times).
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 16
#define MAX_MASK_ELEMS 8192   // 32 KB of the 64 KB constant memory
__constant__ float const_mask[MAX_MASK_ELEMS];

// KT/CT > 0: compile-time kernel size / channel count; 0: use runtime value
template <int KT, int CT>
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const int B, const int M, const int C_rt, const int H, const int W, const int K_rt, const int S)
{
    const int K = KT > 0 ? KT : K_rt;
    const int C = CT > 0 ? CT : C_rt;
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    const int W_grid = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int m = blockIdx.x;
    const int b = blockIdx.z;
    const int h = (blockIdx.y / W_grid) * BLOCK_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_grid) * BLOCK_SIZE + threadIdx.x;
    if (h >= H_out || w >= W_out)
        return;   // safe: this kernel has no __syncthreads()

    const float *in_b = input + (size_t)b * C * H * W + (h * S) * W + (w * S);
    const float *mask_m = const_mask + m * C * K * K;
    float sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < C; c++) {
        #pragma unroll
        for (int p = 0; p < K; p++) {
            #pragma unroll
            for (int q = 0; q < K; q++)
                sum += in_b[(c * H + p) * W + q] * mask_m[(c * K + p) * K + q];
        }
    }
    output[(((size_t)b * M + m) * H_out + h) * W_out + w] = sum;
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
    // the mask lives in __constant__ memory; there is no device buffer for it,
    // but Mini-DNN later passes *device_mask_ptr to epilog -> cudaFree(nullptr) is a no-op
    *device_mask_ptr = nullptr;
    check_cuda("prolog malloc");
    if ((size_t)M * C * K * K > MAX_MASK_ELEMS) {
        std::cout << "mask (" << M * C * K * K << " floats) does not fit in constant memory" << std::endl;
        exit(-1);
    }

    cudaMemcpy(*device_input_ptr, host_input, (size_t)B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_mask, host_mask, (size_t)M * C * K * K * sizeof(float));
    check_cuda("prolog memcpy");
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid = (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int H_grid = (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(M, H_grid * W_grid, B);

    if (K == 7 && C == 1)
        conv_forward_kernel<7, 1><<<gridDim, blockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
    else if (K == 7 && C == 4)
        conv_forward_kernel<7, 4><<<gridDim, blockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
    else if (K == 7)
        conv_forward_kernel<7, 0><<<gridDim, blockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
    else if (K == 3)
        conv_forward_kernel<3, 0><<<gridDim, blockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
    else
        conv_forward_kernel<0, 0><<<gridDim, blockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
    check_cuda("conv_forward_kernel launch");
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
