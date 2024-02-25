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
#define     MASK_SIZE      3
#define     MASK_RADIUS    (MASK_SIZE / 2)
#define     TILE_SIZE      MASK_SIZE 
#define     W	(TILE_SIZE + MASK_SIZE - 1)

////@@ Define constant memory for device kernel here
__constant__ float c_deviceKernel[MASK_SIZE * MASK_SIZE * MASK_SIZE]; 

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
    ////@@ Insert kernel code her
    __shared__ float Nds[W][W][W];
    int tid = threadIdx.x + (threadIdx.y * MASK_SIZE) + (threadIdx.z * MASK_SIZE * MASK_SIZE);

    if (tid < W * W)
    {
	int tileX = tid % W;
	int tileY = (tid / W) % W;
	int srcX = blockIdx.x * TILE_SIZE + tileX - MASK_RADIUS;
	int srcY = blockIdx.y * TILE_SIZE + tileY - MASK_RADIUS;
	int srcZ = blockIdx.z * TILE_SIZE - MASK_RADIUS;
	for (int i = 0; i < W; i++)
	{
	    int zpos = srcZ + i;

	    if(zpos >= 0 && zpos < z_size && srcY >= 0 && srcY < y_size && srcX >= 0 && srcX < x_size)
	    {
		int src = zpos * x_size * y_size + srcY * x_size + srcX;
		Nds[tileX][tileY][i] = input[src];
	    }
	    else
	    {
		Nds[tileX][tileY][i] = 0;
	    }
	}
    }

    __syncthreads();

    float result = 0;
    int z = threadIdx.z + (blockIdx.z * TILE_SIZE);
    int y = threadIdx.y + (blockIdx.y * TILE_SIZE);
    int x = threadIdx.x + (blockIdx.x * TILE_SIZE);
    if(z < z_size && y < y_size && x < x_size)
    {
	for (int i = 0; i < MASK_SIZE; ++i)
	{
	    for (int j = 0; j < MASK_SIZE; ++j)
            {
		for (int k = 0; k < MASK_SIZE; ++k)
		{
		    result += Nds[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k] * c_deviceKernel[k * MASK_SIZE * MASK_SIZE + j * MASK_SIZE + i];
		}
	    }
	}
	output[x + (y * x_size) + (z * x_size * y_size)] = result;
    }

    __syncthreads();
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
