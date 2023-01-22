/*
This programm compute y_i = sin(sqrt(x_i)), where x_i = 2*pi*i/N
*/

#include <stdio.h>

#define N (1024*1024)

__global__ void kernel(float* dArr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float x_i = 2.0f * 3.1415926f * (float)idx / (float)N;

  dArr[idx] = sinf(sqrtf(x_i));
}

int main() {
  float* hArr;
  float* dArr;

  hArr = (float*) malloc(N * (sizeof(float)));
  cudaMalloc((void**) &dArr, N * sizeof(float));

  // we want 512 threads within a block
  kernel <<< N/512, 512 >>> (dArr);

  cudaError_t err = cudaMemcpy(hArr, dArr, N * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }
  for(int idx = 0; idx < 20; idx++) {
    printf("a[%d] = %f\n", idx, hArr[idx]);
  }
  
  free(hArr);
  cudaFree(dArr);

  return 0;
}