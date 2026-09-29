// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Hierarchical (3-phase) Brent-Kung scan:
//   1. scan          : each block scans 2*BLOCK_SIZE elements, writes its total to aux[]
//   2. scan (1 block): scan the per-block totals aux[] -> blockSums[]
//   3. addBlockSums  : add blockSums[b-1] to every element of block b
// Phase 2 runs in one block, so numBlocks <= 2*BLOCK_SIZE (inputs up to 1M elements).

__global__ void addBlockSums(float *blockSums, float *output, int len) {
  int start = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  if (blockIdx.x > 0) {
    if (start < len)
      output[start] += blockSums[blockIdx.x - 1];
    if (start + blockDim.x < len)
      output[start + blockDim.x] += blockSums[blockIdx.x - 1];
  }
}

__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float shared[BLOCK_SIZE * 2];

  int start = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  if (start < len)
    shared[threadIdx.x] = input[start];
  else
    shared[threadIdx.x] = 0.0;
  if (start + blockDim.x < len)
    shared[blockDim.x + threadIdx.x] = input[start + blockDim.x];
  else
    shared[blockDim.x + threadIdx.x] = 0.0;

  // up-sweep (reduction tree)
  for (int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE)
      shared[index] += shared[index - stride];
  }

  // down-sweep (distribution tree)
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE)
      shared[index + stride] += shared[index];
  }
  __syncthreads();

  if (start < len)
    output[start] = shared[threadIdx.x];
  if (start + blockDim.x < len)
    output[start + blockDim.x] = shared[blockDim.x + threadIdx.x];
  if (aux != NULL && threadIdx.x == blockDim.x - 1)
    aux[blockIdx.x] = shared[2 * blockDim.x - 1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *aux;
  float *SumAndScan;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&SumAndScan, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
  if (numBlocks > 2 * BLOCK_SIZE) {
    wbLog(ERROR, "Input too large for a two-level scan: ", numElements);
    return -1;
  }
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 OneDGrid(1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, aux, numElements);
  // scan only the numBlocks valid block totals (was 2*BLOCK_SIZE -> out-of-bounds)
  scan<<<OneDGrid, dimBlock>>>(aux, SumAndScan, NULL, numBlocks);
  addBlockSums<<<dimGrid, dimBlock>>>(SumAndScan, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux);
  cudaFree(SumAndScan);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
