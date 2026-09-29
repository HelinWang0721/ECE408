// op2.cu -- Optimization: Shared memory matrix multiplication and input
//            matrix unrolling (3 points)
//
// Convolution as GEMM (im2col):
//   1. unroll_kernel writes, for each image, the unrolled input matrix
//        X_unroll[C*K*K][H_out*W_out],  X_unroll[c*K*K + p*K + q][h*W_out + w]
//          = x[b][c][h*S + p][w*S + q]
//   2. matmul_shared multiplies the mask, already laid out as a
//        W[M][C*K*K] matrix (mask[m][c][p][q] is row-major M x CKK),
//      with X_unroll using TILE_WIDTH x TILE_WIDTH shared-memory tiles:
//        Y[b][M][H_out*W_out] = W * X_unroll[b]
// gridDim.z indexes the image inside a chunk. Unrolling expands the input
// K*K / S^2 times (49x for the LeNet layers), so images are processed in
// chunks that keep the X_unroll buffer at <= UNROLL_BUDGET floats.
//
// Bugs fixed vs. the previous version: there was no unrolling at all; the
// kernel multiplied the raw input (as an H x W matrix) by the mask (as a
// W x H matrix) and wrote H x W results, ignoring b, m, c, K and S.
#include <cmath>
#include <iostream>
#include <algorithm>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNROLL_THREADS 256
#ifndef UNROLL_BUDGET
#define UNROLL_BUDGET (64 * 1024 * 1024)   // floats (256 MB) for X_unroll
#endif

// One thread per element of X_unroll (for n_img images starting at b0).
__global__ void unroll_kernel(float *x_unroll, const float *input, const int b0, const int n_img, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int HW_out = H_out * W_out;
    const int KK = K * K;
    const long long rows = (long long)C * KK;
    const long long per_img = rows * HW_out;
    const long long total = per_img * n_img;

    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        int img = (int)(idx / per_img);
        long long r = idx % per_img;
        int row = (int)(r / HW_out);      // c*K*K + p*K + q
        int col = (int)(r % HW_out);      // h*W_out + w
        int c = row / KK, p = (row % KK) / K, q = row % K;
        int h = col / W_out, w = col % W_out;
        x_unroll[idx] = input[((long long)(b0 + img) * C + c) * H * W + (h * S + p) * W + (w * S + q)];
    }
}

// Y[z] (numARows x numBCols) = A (numARows x numACols) * B[z] (numACols x numBCols)
__global__ void matmul_shared(const float *A, const float *Bmat, float *Cmat, const int numARows, const int numACols, const int numBCols)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    const float *Bz = Bmat + (long long)blockIdx.z * numACols * numBCols;
    float *Cz = Cmat + (long long)blockIdx.z * numARows * numBCols;

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (numACols + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < numARows && aCol < numACols) ? A[row * numACols + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < numACols && col < numBCols) ? Bz[(long long)bRow * numBCols + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < numARows && col < numBCols)
        Cz[(long long)row * numBCols + col] = acc;
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
    const int HW_out = H_out * W_out;
    const int CKK = C * K * K;

    // how many images fit into the unroll buffer
    const long long per_img = (long long)CKK * HW_out;
    const int chunk = (int)std::max(1LL, std::min((long long)B, (long long)UNROLL_BUDGET / per_img));

    float *x_unroll;
    cudaMalloc((void **)&x_unroll, (size_t)chunk * per_img * sizeof(float));
    check_cuda("malloc x_unroll");

    for (int b0 = 0; b0 < B; b0 += chunk) {
        int n = std::min(chunk, B - b0);
        long long total = per_img * n;
        int blocks = (int)std::min((total + UNROLL_THREADS - 1) / UNROLL_THREADS, 65535LL * 8);
        unroll_kernel<<<blocks, UNROLL_THREADS>>>(x_unroll, device_input, b0, n, C, H, W, K, S);

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 gridDim((HW_out + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, n);
        matmul_shared<<<gridDim, blockDim>>>(device_mask, x_unroll, device_output + (size_t)b0 * M * HW_out, M, CKK, HW_out);
    }
    check_cuda("unroll/matmul launch");
    cudaFree(x_unroll);
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
