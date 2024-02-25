#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define BLOCK_SIZE 32

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    // Allocate shared memory for input and mask tiles
    __shared__ float input_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float mask_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float output_value = 0.0;

    // Loop over the mask and input tiles to perform matrix multiplication
    for (int tile_idx = 0; tile_idx < (W - 1) / BLOCK_SIZE + 1; ++tile_idx) {

        // Load input tile into shared memory
        if (row < H && tile_idx * BLOCK_SIZE + threadIdx.x < W) {
            input_tile[threadIdx.y][threadIdx.x] = input[row * W + tile_idx * BLOCK_SIZE + threadIdx.x];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load mask tile into shared memory
        if (col < W && tile_idx * BLOCK_SIZE + threadIdx.y < H) {
            mask_tile[threadIdx.y][threadIdx.x] = mask[col * H + tile_idx * BLOCK_SIZE + threadIdx.y];
        } else {
            mask_tile[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Unrolled loop for matrix multiplication
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            output_value += mask_tile[threadIdx.y][k] * input_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

        // Write the computed value to the output matrix
    if (row < H && col < W) {
        output[row * W + col] = output_value;
    }
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    
    cudaMalloc((void **)device_output_ptr, B * M * ((H - K)/S + 1) *  ((W - K)/S + 1) * sizeof(float));
    cudaMalloc((void **)device_input_ptr, B * C *H * W * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float));
    cudaError_t error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int W_gride = ceil((float)((W - K) / S + 1) / BLOCK_SIZE);
    int H_gride = ceil((float)((H - K) / S + 1) / BLOCK_SIZE);
    int Y_gride = W_gride * H_gride;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(M, Y_gride, B);
    
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, B * M * ((H - K)/S + 1) * ((W - K)/S + 1)* sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
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
