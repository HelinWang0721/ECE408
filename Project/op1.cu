#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define BLOCK_SIZE 32


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
   // Allocate shared memory
   __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ float M_ds[BLOCK_SIZE][BLOCK_SIZE];

   // Calculate global row and column indices
   int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
   int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

   // Perform tiling
   for (int m = 0; m < (BLOCK_SIZE + K - 1) / BLOCK_SIZE; m++) {
       if (m * BLOCK_SIZE + threadIdx.x < K && Row < H) {
           M_ds[threadIdx.y][threadIdx.x] = mask[Row * K + m * BLOCK_SIZE + threadIdx.x];
       }
       else {
           M_ds[threadIdx.y][threadIdx.x] = 0.0;
       }

       if (m * BLOCK_SIZE + threadIdx.y < K && Col < W) {
           N_ds[threadIdx.y][threadIdx.x] = input[Col * K + m * BLOCK_SIZE + threadIdx.y];
       }
       else {
           N_ds[threadIdx.y][threadIdx.x] = 0.0;
       }

       __syncthreads();

       // Convolution computation using tiles
       for (int k = 0; k < BLOCK_SIZE; ++k) {
           for (int i = 0; i < K; ++i) {
               for (int j = 0; j < K; ++j) {
                   output[Row * W + Col] += M_ds[k][i] * N_ds[k][j];
               }
           }
       }
       __syncthreads();
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
