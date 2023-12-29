
//modulo operation removed, bank conflicts may occur
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 256

__global__ void sumReduction (int *v, int *v_r) {
  __shared__ int partialSum[SHMEM_SIZE];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  partialSum[threadIdx.x] = v[tid];
  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) { //reduce the number of threads in each iteration
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x) {
      partialSum[index] += partialSum[index + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
  v_r[blockIdx.x] = partialSum[0];
  }
}

int main() {
  int N = 1 << 16;
  size_t bytes = N * sizeof(int);

  vector<int>h_v(N);
  vector<int>h_v_r(N);

  generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

  int *d_v, *d_v_r;
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

  const int TB_SIZE = 256;

  int GRID_SIZE = N / TB_SIZE;

  sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
  sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

  cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

  assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

  cout << "Done";

  return 0;
}
