// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
// Kernel configuration
#define BLOCK_SIZE 16

// Greyscale conversion kernel
// Converts RGB input image to greyscale in dBuffer
// Also copies image to device for later use 
__global__ void getGreyScale(float *dInput, unsigned char* dBuffer, unsigned char* dGrey, int width, int height, int channels){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  int y = blockIdx.y * BLOCK_SIZE + ty;

  int i = x * height + y;

  if(x < width && y < height){
    unsigned char r  = (unsigned char) (255 * dInput[3*i]);
    unsigned char g = (unsigned char) (255 * dInput[3*i + 1]);
    unsigned char b = (unsigned char) (255 * dInput[3*i + 2]);

    dBuffer[3 * i] = r;
    dBuffer[3 * i + 1] = g;
    dBuffer[3 * i + 2] = b;

    dGrey[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
  __syncthreads();
}



float p(int x, int size){
  return (float) x / (float) size;
}


// Histogram computation kernel
// Calculates histogram of greyscale image in dGrey
// Histogram stored in device histogram
__global__ void getHist(unsigned char* dGrey, int* histogram, int width, int height){
  int tx = threadIdx.x;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  // int stride = width;
  for(int y = 0; y < height; y++){
    int i = x * height + y;
    if(x < width && y < height){
      atomicAdd(&(histogram[dGrey[i]]), 1);
    }
  }
}

__global__ void getColor(unsigned char* dBuffer, float *deviceOutput, float *cdf, float cdfmin, int width, int height){
  int ty = threadIdx.y;
  int tx = threadIdx.x;


  int y = blockIdx.y * BLOCK_SIZE + ty;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  int i = x * height + y;
  

  // Check bounds
  if(x < width && y < height){

    // Process each color channel  
    for(int c = 0; c < 3; c++){

      // Get color channel pixel index
      int idx = 3 * i + c;

      // Use CDF to map original value to new value
      float new_color = 255.0*(cdf[dBuffer[idx]] - cdfmin)/(1.0 - cdfmin);

      // Clamp value to valid range
      new_color = max(new_color, 0.0);
      new_color = min(new_color, 255.0);

      // Write output
      deviceOutput[idx] = (float)(new_color/255.0);

    }

    __syncthreads();
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
  float *dInput;
  unsigned char* dBuffer;
  unsigned char* dGrey;
  float *deviceOutput;
  int *histogram;
  int *host_histogram;
  float *cdf;
  float *host_cdf;
  
  args = wbArg_read(argc, argv); 
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

  host_histogram = (int *) malloc(256 * sizeof(int));
  host_cdf = (float *) malloc(256 * sizeof(float));  
  int sInput = imageWidth * imageHeight * imageChannels * sizeof(float);
  int sBuffer = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);
  int sizeGrey = imageWidth * imageHeight * sizeof(unsigned char);
  int sizeHis = 256 * sizeof(int);
  int sizeCDF = 256 * sizeof(float);


  cudaMalloc((void **)&dInput, sInput);
  cudaMalloc((void **)&dBuffer, sBuffer);
  cudaMalloc((void **)&dGrey, sizeGrey);
  cudaMalloc((void **)&histogram, sizeHis);
  cudaMalloc((void **)&cdf, sizeCDF);
  cudaMalloc((void **)&deviceOutput, sInput);


  cudaMemcpy(dInput, hostInputImageData, sInput, cudaMemcpyHostToDevice);

// Grid configuration
  // Based on image dims and block size
  dim3 dimGrid(ceil((1.0 * imageWidth) / BLOCK_SIZE), ceil((1.0 * imageHeight) / BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
// Greyscale kernel launch
  getGreyScale<<<dimGrid, dimBlock>>>(dInput, dBuffer, dGrey, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // 1D grid for histogram
  dim3 dimGrid2(ceil((1.0 * imageWidth) / BLOCK_SIZE), 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  getHist<<<dimGrid2, dimBlock2>>>(dGrey, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // Copy data, compute CDF, etc.

  // Color correction launch
  cudaMemcpy(host_histogram, histogram, sizeHis, cudaMemcpyDeviceToHost);

  int imagesize = imageWidth * imageHeight;
  host_cdf[0] = p(host_histogram[0], imagesize);
  for (int i = 1; i < 256; i++){
    host_cdf[i] = host_cdf[i - 1] + p(host_histogram[i], imagesize);
  }
  cudaMemcpy(cdf, host_cdf, sizeCDF, cudaMemcpyHostToDevice);
  getColor<<<dimGrid, dimBlock>>>(dBuffer, deviceOutput, cdf, host_cdf[0], imageWidth, imageHeight);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutput, sInput, cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);
  // Free memory
  cudaFree(dGrey);
  cudaFree(histogram);
  cudaFree(cdf);
  cudaFree(dInput);
  cudaFree(dBuffer);
  cudaFree(deviceOutput);

  free(host_histogram);
  free(host_cdf);
  return 0;
}