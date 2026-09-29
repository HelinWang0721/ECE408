// op5.cu -- Stacked optimizations on top of op4 (tree channel reduction):
//   * Weight matrix (kernel values) in constant memory (0.5 point)
//       every thread of a warp reads the same mask element at the same time
//       -> one broadcast from the constant cache instead of global loads
//   * Tuning with restrict and loop unrolling (3 points)
//       __restrict__ promises input/output do not alias so the compiler can
//       keep loads in registers / use the read-only path; K is a template
//       parameter for the LeNet layers (K = 7) and the test cases (K = 3) so
//       `#pragma unroll` fully unrolls the K x K loops (no loop counters,
//       constant offsets, more ILP). Other K values use the generic kernel.
//
// Bugs fixed vs. the previous version: BLOCK_SIZE 128 -> 128*128 = 16384
// threads per block (limit is 1024, the launch always failed), plus all the
// tree-reduction bugs listed in op4.cu, and cudaFree() of a mask pointer that
// was never allocated.
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define MAX_MASK_ELEMS 8192   // 32 KB of the 64 KB constant memory
__constant__ float const_mask[MAX_MASK_ELEMS];

// KT > 0: compile-time kernel size (fully unrolled); KT == 0: runtime K
template <int KT>
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K_rt, const int S)
{
    const int K = KT > 0 ? KT : K_rt;
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int T = blockDim.x;
    const int CZ = blockDim.z;

    extern __shared__ float partial[];   // [CZ][T][T]

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define tree(z, y, x) partial[((z) * T + (y)) * T + (x)]

    const int W_grid = (W_out + T - 1) / T;
    const int m = blockIdx.x;
    const int b = blockIdx.z;
    const int h = (blockIdx.y / W_grid) * T + threadIdx.y;
    const int w = (blockIdx.y % W_grid) * T + threadIdx.x;
    const bool valid = (h < H_out && w < W_out);

    float sum = 0.0f;
    if (valid) {
        for (int c = threadIdx.z; c < C; c += CZ) {
            const float *in_c = input + ((size_t)b * C + c) * H * W + (h * S) * W + (w * S);
            const float *mask_c = const_mask + (m * C + c) * K * K;
            #pragma unroll
            for (int p = 0; p < K; p++) {   // K is a compile-time constant when KT > 0
                #pragma unroll
                for (int q = 0; q < K; q++)
                    sum += in_c[p * W + q] * mask_c[p * K + q];
            }
        }
    }
    tree(threadIdx.z, threadIdx.y, threadIdx.x) = sum;

    for (int stride = CZ / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.z < stride)
            tree(threadIdx.z, threadIdx.y, threadIdx.x) += tree(threadIdx.z + stride, threadIdx.y, threadIdx.x);
    }

    if (threadIdx.z == 0 && valid)
        out_4d(b, m, h, w) = tree(0, threadIdx.y, threadIdx.x);

    #undef out_4d
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

    int CZ = 1;
    while (CZ < C && CZ < 64) CZ *= 2;
    int T = 16;
    while (T > 1 && T * T * CZ > 1024) T /= 2;

    const int W_grid = (W_out + T - 1) / T;
    const int H_grid = (H_out + T - 1) / T;
    dim3 blockDim(T, T, CZ);
    dim3 gridDim(M, H_grid * W_grid, B);
    const size_t smem = (size_t)T * T * CZ * sizeof(float);

    if (K == 7)
        conv_forward_kernel<7><<<gridDim, blockDim, smem>>>(device_output, device_input, B, M, C, H, W, K, S);
    else if (K == 3)
        conv_forward_kernel<3><<<gridDim, blockDim, smem>>>(device_output, device_input, B, M, C, H, W, K, S);
    else
        conv_forward_kernel<0><<<gridDim, blockDim, smem>>>(device_output, device_input, B, M, C, H, W, K, S);
    check_cuda("op5 conv launch");
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
