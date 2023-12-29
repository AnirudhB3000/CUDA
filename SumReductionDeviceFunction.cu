//modulo operation removed, bank conflicts avoided, idle threads reduced, device function is used in last iteration to save useless work
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

#define SIZE 256
#define SHMEM_SIZE 256 * 4


__device__ void warpReduce(volatile int *shmem_ptr, int t) {
  shmem_ptr[t] += shmem_ptr[t + 32];
  shmem_ptr[t] += shmem_ptr[t + 16];
  shmem_ptr[t] += shmem_ptr[t + 8];
  shmem_ptr[t] += shmem_ptr[t + 4];
  shmem_ptr[t] += shmem_ptr[t + 2];
  shmem_ptr[t] += shmem_ptr[t + 1];
}
__global__ void sumReduction (int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];

  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    // Each thread does work unless it is further than the stride
    if (threadIdx.x < s) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x < 32) {
    warpReduce(partial_sum, threadIdx.x);
  }

  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
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

  const int TB_SIZE = SIZE;

  int GRID_SIZE = N / TB_SIZE / 2;

  sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
  sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

  cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

  assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

  cout << "Done";

  return 0;
}