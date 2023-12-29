#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__ void MatrixMultiplication(const int *a, const int *b, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  __shared__ int s_a[SHMEM_SIZE]; //static alloc
  __shared__ int s_b[SHMEM_SIZE];

  int tmp = 0;

  for (int i = 0; i < N; i += blockDim.x) {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x]; //load elements into tile
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];


    __syncthreads(); //sync all async threads


    for (int j = 0; j < blockDim.x; j++) { // main loop to multiply
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }


    __syncthreads(); // sync all async threads
  }

  c[row * N + col] = tmp;
}

void verify_result(vector <int> &a, vector <int> &b, vector <int> &c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int temp = 0;
      for (int k = 0; k < N; k++) {
        temp += a[i * N + k] * b[k * N + j];
      }
      assert(temp == c[i * N + j]);
    }
  }
}

int main() {
  int N = 1 << 10;
  size_t bytes = N * N * sizeof(int);

  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() {return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() {return rand() % 100; });

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int THREADS = 32;
  int BLOCKS = N/ THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  MatrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(h_a, h_b, h_c, N);

  cout << "Done";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}