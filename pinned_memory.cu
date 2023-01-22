/*
This programm is an example of working with pinned memory
based on summation of two arrays:
a_i = 1/(i + 1)^2
b_i = exp(1/(i + 1))
*/

#include <stdio.h>

__global__ void function(float* dA, float* dB, float* dC, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    dC[idx] = dA[idx] + dB[idx];
  }
}

int main() {
  float timerValueGPU, timerValueCPU;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *hA, *hB, *hC, *dA, *dB, *dC;
  int size = 512 * 50000; // size of each array
  int N_thread = 512; // number of threads within a block
  int N_blocks;

  // create arrays on host
  unsigned int mem_size = sizeof(float) * size;
  hA = (float*) malloc(mem_size);
  hB = (float*) malloc(mem_size);
  hC = (float*) malloc(mem_size);

  // create arrays on device
  cudaMalloc((void**) &dA, mem_size);
  cudaMalloc((void**) &dB, mem_size);
  cudaMalloc((void**) &dC, mem_size);

  // filling the arrays
  for (int i = 0; i < size; i++) {
    hA[i] = 1.0 / ((i + 1) * (i + 1));
    hB[i] = expf(1.0 / (i + 1));
    hC[i] = 0.0;
  }

  // calculating number of blocks
  if ((size % N_thread) == 0) {
    N_blocks = size / N_thread;
  }
  else {
    N_blocks = (int) (size / N_thread) + 1;
  }

  dim3 blocks(N_blocks);

  // GPU variant
  cudaEventRecord(start, 0);

  cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);
  
  function <<< N_blocks, N_thread >>> (dA, dB, dC, size);

  cudaError_t err = cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueGPU, start, stop);

  printf("\nGPU time: %f ms\n", timerValueGPU);

  // CPU variant
  cudaEventRecord(start, 0);

  for (int i = 0; i < size; i++) {
    hC[i] = hA[i] + hB[i];
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timerValueCPU, start, stop);

  printf("\nCPU time: %f ms\n", timerValueCPU);
  printf("Rate: %f x\n", timerValueCPU / timerValueGPU);
  
  return 0;
}














