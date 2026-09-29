#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

////@@ Define any useful program-wide constants here
// Strategy: output tile TILE_SIZE^3 per block (one thread per output voxel),
// input tile (TILE_SIZE + MASK_SIZE - 1)^3 in shared memory, mask in constant memory.
// (Previously TILE_SIZE == MASK_SIZE == 3: 27-thread blocks whose loaded data
//  was ~78% halo, and the load index hard-coded MASK_SIZE as the block width.)
#define     MASK_SIZE      3
#define     MASK_RADIUS    (MASK_SIZE / 2)
#define     TILE_SIZE      8
#define     W	(TILE_SIZE + MASK_SIZE - 1)

////@@ Define constant memory for device kernel here
__constant__ float c_deviceKernel[MASK_SIZE * MASK_SIZE * MASK_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
    ////@@ Insert kernel code her
    // [z][y][x]: consecutive threadIdx.x -> consecutive shared addresses (no bank conflicts)
    __shared__ float Nds[W][W][W];
    int tid = threadIdx.x + threadIdx.y * TILE_SIZE + threadIdx.z * TILE_SIZE * TILE_SIZE;

    // cooperative load of the W^3 input tile (incl. halo, zero-padded ghost cells)
    for (int i = tid; i < W * W * W; i += TILE_SIZE * TILE_SIZE * TILE_SIZE) {
        int tileX = i % W;
        int tileY = (i / W) % W;
        int tileZ = i / (W * W);
        int srcX = blockIdx.x * TILE_SIZE + tileX - MASK_RADIUS;
        int srcY = blockIdx.y * TILE_SIZE + tileY - MASK_RADIUS;
        int srcZ = blockIdx.z * TILE_SIZE + tileZ - MASK_RADIUS;
        if (srcZ >= 0 && srcZ < z_size && srcY >= 0 && srcY < y_size && srcX >= 0 && srcX < x_size)
            Nds[tileZ][tileY][tileX] = input[(srcZ * y_size + srcY) * x_size + srcX];
        else
            Nds[tileZ][tileY][tileX] = 0.0f;
    }
    __syncthreads();

    int z = threadIdx.z + blockIdx.z * TILE_SIZE;
    int y = threadIdx.y + blockIdx.y * TILE_SIZE;
    int x = threadIdx.x + blockIdx.x * TILE_SIZE;
    if (z < z_size && y < y_size && x < x_size) {
        float result = 0.0f;
        for (int k = 0; k < MASK_SIZE; ++k)          // z
            for (int j = 0; j < MASK_SIZE; ++j)      // y
                for (int i = 0; i < MASK_SIZE; ++i)  // x
                    result += Nds[threadIdx.z + k][threadIdx.y + j][threadIdx.x + i] *
                              c_deviceKernel[(k * MASK_SIZE + j) * MASK_SIZE + i];
        output[(z * y_size + y) * x_size + x] = result;
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  ////@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  ////@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_deviceKernel, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  ////@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size * 1.0/TILE_SIZE), ceil(y_size * 1.0 / TILE_SIZE), ceil(z_size* 1.0/TILE_SIZE));
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  ////@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  ////@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float),cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
