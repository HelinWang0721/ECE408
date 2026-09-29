// op1.cu -- Optimization: Tiled shared memory convolution (2 points)
//
// Each block computes a TILE_WIDTH x TILE_WIDTH tile of one output feature map
// (m) of one image (b). For every input channel c it first cooperatively
// loads the input patch this tile needs -- (TILE_WIDTH-1)*S + K pixels per side,
// i.e. the tile plus its halo -- and the K x K mask slice into shared memory.
// It then computes from shared memory. Each input pixel is read from global
// memory once per block instead of up to K*K times.
//
// Launch: grid(M, H_grid * W_grid, B), block(TILE_WIDTH, TILE_WIDTH).
// Dynamic shared memory: (IN_TILE^2 + K^2) floats.
//
// Bugs fixed vs. the previous version: it indexed input/mask as if the problem
// were a matrix multiply (mask[Row*K + ...], input[Col*K + ...]), ignored
// b/m/c/stride, accumulated with += into uninitialised global output and
// wrote output[Row*W + Col] far outside the output tensor.
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int IN_TILE = (TILE_WIDTH - 1) * S + K;  // input patch side incl. halo

    extern __shared__ float smem[];
    float *in_s = smem;                         // IN_TILE * IN_TILE
    float *mask_s = smem + IN_TILE * IN_TILE;   // K * K

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int m = blockIdx.x;
    const int b = blockIdx.z;
    const int h0 = (blockIdx.y / W_grid) * TILE_WIDTH;   // first output row of this tile
    const int w0 = (blockIdx.y % W_grid) * TILE_WIDTH;   // first output col of this tile
    const int h = h0 + threadIdx.y;
    const int w = w0 + threadIdx.x;
    const int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    const int nthreads = TILE_WIDTH * TILE_WIDTH;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        // cooperative load of the input patch (all threads take part, even
        // those whose output pixel is out of range -- no early return before
        // a barrier)
        for (int i = tid; i < IN_TILE * IN_TILE; i += nthreads) {
            int r = h0 * S + i / IN_TILE;
            int col = w0 * S + i % IN_TILE;
            in_s[i] = (r < H && col < W) ? in_4d(b, c, r, col) : 0.0f;
        }
        for (int i = tid; i < K * K; i += nthreads)
            mask_s[i] = mask_4d(m, c, i / K, i % K);
        __syncthreads();

        if (h < H_out && w < W_out) {
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    acc += in_s[(threadIdx.y * S + p) * IN_TILE + threadIdx.x * S + q] * mask_s[p * K + q];
        }
        __syncthreads();  // before the next channel overwrites the tiles
    }
    if (h < H_out && w < W_out)
        out_4d(b, m, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int IN_TILE = (TILE_WIDTH - 1) * S + K;
    const size_t smem = (size_t)(IN_TILE * IN_TILE + K * K) * sizeof(float);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, H_grid * W_grid, B);
    conv_forward_kernel<<<gridDim, blockDim, smem>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
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
