// Histogram Equalization
//
// Pipeline (all on the GPU):
//   1. castAndGreyScale : float RGB -> uchar RGB (kept for step 5) + uchar grey
//   2. histogram        : privatized (shared-memory) histogram, atomics
//   3. cdfScan          : one-block Kogge-Stone scan of the 256 bins -> CDF
//   4. equalize         : ucharImage -> correct_color() -> float output
// cdfmin is cdf[0] because the CDF is monotonically non-decreasing.

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Step 1: one thread per pixel. Pixel index is row-major (y * width + x)
// so consecutive threads touch consecutive addresses (coalesced).
__global__ void castAndGreyScale(const float *input, unsigned char *ucharImage,
                                 unsigned char *grey, int numPixels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numPixels) {
    unsigned char r = (unsigned char)(255 * input[3 * i]);
    unsigned char g = (unsigned char)(255 * input[3 * i + 1]);
    unsigned char b = (unsigned char)(255 * input[3 * i + 2]);
    ucharImage[3 * i]     = r;
    ucharImage[3 * i + 1] = g;
    ucharImage[3 * i + 2] = b;
    // NOTE: the README pseudo-code truncates, but the expected outputs in
    // data/*/output.ppm were generated with a *rounded* grey value (verified
    // against all 10 datasets), hence the +0.5.
    grey[i] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b + 0.5);
  }
}

// Step 2: privatization. Each block accumulates into a shared-memory
// histogram (fast shared atomics, little contention on global memory),
// then merges its 256 bins into the global histogram once.
__global__ void histogram(const unsigned char *grey, unsigned int *hist, int numPixels) {
  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];
  for (int bin = threadIdx.x; bin < HISTOGRAM_LENGTH; bin += blockDim.x)
    histo_s[bin] = 0;
  __syncthreads();

  // grid-stride loop: works for any grid size
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numPixels; i += blockDim.x * gridDim.x)
    atomicAdd(&histo_s[grey[i]], 1u);
  __syncthreads();

  for (int bin = threadIdx.x; bin < HISTOGRAM_LENGTH; bin += blockDim.x)
    if (histo_s[bin] > 0)
      atomicAdd(&hist[bin], histo_s[bin]);
}

// Step 3: inclusive Kogge-Stone scan of 256 values in a single block
// (launch with exactly HISTOGRAM_LENGTH threads).
__global__ void cdfScan(const unsigned int *hist, float *cdf, int numPixels) {
  __shared__ float s[HISTOGRAM_LENGTH];
  int t = threadIdx.x;
  s[t] = hist[t] / (float)numPixels;  // p(x)
  for (int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2) {
    __syncthreads();
    float v = (t >= stride) ? s[t - stride] : 0.0f;
    __syncthreads();  // everyone has read before anyone writes
    s[t] += v;
  }
  cdf[t] = s[t];
}

// Step 4: apply correct_color() and cast back to float.
__global__ void equalize(const unsigned char *ucharImage, const float *cdf, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float cdfmin = cdf[0];
    float v = 255.0f * (cdf[ucharImage[i]] - cdfmin) / (1.0f - cdfmin);
    v = fminf(fmaxf(v, 0.0f), 255.0f);  // clamp(x, 0, 255)
    unsigned char corrected = (unsigned char)v;  // the reference stores it back into ucharImage
    output[i] = (float)(corrected / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  float *deviceOutput;
  unsigned char *deviceUchar;
  unsigned char *deviceGrey;
  unsigned int *deviceHist;
  float *deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int numPixels = imageWidth * imageHeight;
  int numValues = numPixels * imageChannels;  // channels == 3 (RGB)

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numValues * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numValues * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUchar, numValues * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrey, numPixels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHist, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  // cudaMalloc does not zero memory -- the histogram must start at 0
  wbCheck(cudaMemset(deviceHist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(Copy, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInputImageData, numValues * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  int pixelBlocks = (numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int valueBlocks = (numValues + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castAndGreyScale<<<pixelBlocks, BLOCK_SIZE>>>(deviceInput, deviceUchar, deviceGrey, numPixels);
  // a modest number of blocks, each thread handles several pixels
  histogram<<<(pixelBlocks < 64 ? pixelBlocks : 64), BLOCK_SIZE>>>(deviceGrey, deviceHist, numPixels);
  cdfScan<<<1, HISTOGRAM_LENGTH>>>(deviceHist, deviceCDF, numPixels);
  equalize<<<valueBlocks, BLOCK_SIZE>>>(deviceUchar, deviceCDF, deviceOutput, numValues);
  wbCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutput, numValues * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceUchar);
  cudaFree(deviceGrey);
  cudaFree(deviceHist);
  cudaFree(deviceCDF);

  return 0;
}
