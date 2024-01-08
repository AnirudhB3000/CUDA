#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;

constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

__global__ void Histogram(char *a, int *result, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int alpha;
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha = a[i] - 'a';
    atomicAdd(&result[alpha / DIV], 1);
  }
}

int main() {
  int N = 1 << 24;

  vector<char> h_input(N);

  vector<int> h_result(BINS);

  srand(1);
  generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

  char *d_input;
  int *d_result;
  cudaMalloc(&d_input, N);
  cudaMalloc(&d_result, BINS * sizeof(int));

  cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int),
             cudaMemcpyHostToDevice);

  int THREADS = 512;

  int BLOCKS = N / THREADS;

  Histogram<<<BLOCKS, THREADS>>>(d_input, d_result, N);

  cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int),
             cudaMemcpyDeviceToHost);

  assert(N == accumulate(begin(h_result), end(h_result), 0));

  ofstream output_file;
  output_file.open("histogram.dat", ios::out | ios::trunc);
  for (auto i : h_result) {
    output_file << i << " \n\n";
  }
  output_file.close();

  cudaFree(d_input);
  cudaFree(d_result);

  return 0;
}