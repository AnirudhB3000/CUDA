#include <cassert>
#include <cstdlib>
#include <iostream>

#define MASK_LENGTH 7

__constant__ int mask[MASK_LENGTH];

__global__ void convolution(int *array, int *result, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int s_array[];

  s_array[threadIdx.x] = array[tid];

  __syncthreads();

  int temp = 0;

  for (int j = 0; j < MASK_LENGTH; j++) {
    if ((threadIdx.x + j) >= blockDim.x) {
      temp += array[tid + j] * mask[j];

    } else {
      temp += s_array[threadIdx.x + j] * mask[j];
    }
  }

  result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}

int main() {
  int n = 1 << 20;
  int bytes_n = n * sizeof(int);
  size_t bytes_m = MASK_LENGTH * sizeof(int);

  int r = MASK_LENGTH / 2;
  int n_p = n + r * 2;

  size_t bytes_p = n_p * sizeof(int);

  int *h_array = new int[n_p];

  for (int i = 0; i < n_p; i++) {
    if ((i < r) || (i >= (n + r))) {
      h_array[i] = 0;
    } else {
      h_array[i] = rand() % 100;
    }
  }

  int *h_mask = new int[MASK_LENGTH];
  for (int i = 0; i < MASK_LENGTH; i++) {
    h_mask[i] = rand() % 10;
  }

  int *h_result = new int[n];

  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p);
  cudaMalloc(&d_result, bytes_n);

  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  int THREADS = 256;

  int GRID = (n + THREADS - 1) / THREADS;

  size_t SHMEM = (THREADS + r * 2) * sizeof(int);

  convolution<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  verify_result(h_array, h_mask, h_result, n);

  std::cout << "Done\n";

  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_result);

  return 0;
}