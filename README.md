# GPU Code Repository

This repository contains GPU-accelerated code for various parallel computing tasks. Each code file addresses a specific task, and this README.md provides guidance on running the programs and lists the necessary requirements.

## How to Run the Programs

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Compiler with CUDA support (e.g., nvcc)

### Running the Programs Locally
1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/gpu-code-repo.git
   ```

2. Navigate to the repository:

   ```bash
   cd gpu-code-repo
   ```

3. Compile the CUDA code:

   ```bash
   nvcc -o executable_name file_name.cu
   ```

4. Run the executable:

   ```bash
   ./executable_name
   ```

### Running the Programs on Google Colab
1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)

2. Upload the CUDA code files (e.g., VectorAdditionBaseline.cu) to your Colab workspace.

3. Create a new Colab notebook and copy the following code snippet to install CUDA and compile the code:

   ```python
   # Install CUDA Toolkit
   !apt update -qq && apt install -y -qq nvidia-cuda-toolkit

   #Write the program with the header:
   %%writefile FileName.cu

   # Compile CUDA code
   !nvcc -o executable_name file_name.cu
   ```

4. Run the compiled executable:

   ```python
   !./executable_name
   ```

   Ensure you have set the runtime type to GPU in Colab: **Runtime -> Change runtime type -> GPU**.

## VectorAdditionBaseline.cu

This CUDA program serves as an introduction to GPU programming by demonstrating a basic vector addition kernel. It leverages parallelism to perform element-wise addition of two vectors. The kernel is executed by multiple threads in parallel, with each thread responsible for adding corresponding elements of the vectors. Memory is allocated both on the device (GPU) and host (CPU), and explicit memory transfers occur between the two.

## VectorAdditionUnified.cu

In this CUDA program, Unified Memory is employed to simplify memory management. Unified Memory allows for a single memory space accessible by both the CPU and GPU. This eliminates the need for explicit data transfers, streamlining the programming model. The vector addition operation remains the same, but memory management is simplified, enhancing code readability and ease of use.

## MatMul.cu

This CUDA code tackles matrix multiplication, a computationally intensive task suitable for parallelization on the GPU. The program provides a basic matrix multiplication kernel, distributing the workload across multiple threads. The kernel exploits parallelism to compute matrix multiplication efficiently.

## MatMulTiled.cu

To optimize matrix multiplication further, this program introduces shared memory tiling. Tiling involves breaking down the matrices into smaller tiles, loading them into shared memory, and performing computations with reduced global memory accesses. This strategy enhances memory access patterns, reducing latency and improving overall performance.

## SumReductionWorkDivergence.cu

The sum reduction code demonstrates a common challenge in parallel computing – work divergence. Due to irregularities in the data, threads might not contribute equally to the reduction, leading to inefficiencies. The code serves as a lesson in managing work divergence and improving thread coordination.

## SumReductionBankConflicts.cu

Bank conflicts can hinder parallel processing efficiency. This program addresses such conflicts in the context of sum reduction by modifying memory access patterns. By optimizing memory access, the code mitigates conflicts, improving the overall performance of the reduction operation.

## SumReductionNoConflicts.cu

Building upon the previous code, this program takes a step further in optimizing sum reduction by entirely eliminating bank conflicts. It focuses on refining memory access patterns to enhance parallelism and reduce contention for shared memory resources.

## SumReductionReducedThreads.cu

In this sum reduction optimization, the program adjusts the number of threads participating in the computation. The goal is to minimize idle threads, ensuring that each thread contributes meaningfully to the reduction operation. This adjustment enhances the overall efficiency of parallel processing.

## SumReductionDeviceFunction.cu

Introducing a device function, this program showcases the use of additional functions to optimize complex kernel logic. By employing a device function, unnecessary work in the last iteration of the sum reduction is eliminated, contributing to a more streamlined and efficient implementation.

## SumReductionCoopGrp.cu

This CUDA code takes advantage of cooperative groups, a feature introduced in CUDA 9.0, to improve thread coordination in sum reduction. Cooperative groups enhance synchronization among threads, reducing overhead and enhancing parallel processing efficiency.

## Convolution.cu

Focusing on a fundamental operation in machine learning, this CUDA program implements a basic convolution operation on a 1D array. Convolution is widely used in Convolutional Neural Networks (CNNs) for tasks like image processing. The code leverages parallelism to enhance the performance of the convolution operation, showcasing the GPU's capabilities in accelerating such computations.

## ConstMemConvolution.cu

Constant memory in CUDA is a specialized region that is cached on the device and read-only for all threads within a thread block. This program utilizes constant memory to store the convolution mask, a set of fixed coefficients used in the convolution operation. By storing the mask in constant memory, every thread within a block can efficiently access the same set of coefficients without redundant fetches from global memory. This optimizes memory access patterns, reducing latency and improving the overall performance of the convolution operation on a 1D array. Constant memory is particularly beneficial when dealing with values that remain constant during the execution of a kernel, making it an ideal choice for storing unchanging parameters like the convolution mask.

## TiledConvolution.cu

This CUDA code performs 1D convolution on a GPU using shared memory optimization for enhanced performance. It slides a constant kernel over an input array in parallel, storing results in a separate array. The code verifies the GPU-computed result against a CPU-computed one. Memory is dynamically allocated, and the convolution mask is stored in the GPU's constant memory. The code demonstrates efficient parallelization of convolution operations, crucial for tasks like signal processing and machine learning.

## CacheConvolution.cu

This CUDA code performs 1D convolution on a GPU with shared memory optimization for improved performance. The convolution kernel slides over an input array, and each thread loads a segment of the array into shared memory, reducing global memory accesses. The convolution operation is then carried out on the shared memory, with proper handling of edge cases to ensure accurate results. The code dynamically allocates memory, utilizes constant memory for the convolution mask, and verifies the GPU-computed result against a CPU-computed one. It showcases efficient parallelization of convolution, essential for applications like signal processing and machine learning.

## 2DConvolution.cu

This CUDA code performs 2D convolution on a square matrix using a given convolution mask. The convolution operation is parallelized on the GPU to enhance performance. The convolution kernel is applied to each element of the output matrix, taking into account the neighboring elements and the convolution mask. The convolution is performed using a two-dimensional grid of thread blocks and threads.
The code includes functions to initialize the input matrix and the convolution mask with random values. It allocates memory on both the host and device, transfers data between them, and launches the convolution kernel. The computed result is then transferred back to the host, and a verification step ensures the correctness of the GPU-computed result against a CPU-computed one.
The CUDA code is structured to efficiently handle 2D convolution tasks, commonly used in image processing, computer vision, and deep learning. The use of shared memory and constant memory for the convolution mask contributes to optimized memory access patterns, improving overall performance. The code is designed to work with matrices of varying sizes, providing flexibility for different applications.

**Note:** Ensure you have the necessary dependencies and a compatible GPU before running these CUDA programs. Adjust the compilation and execution commands based on your system configuration.