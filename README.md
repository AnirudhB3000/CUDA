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

This code performs element-wise addition of two vectors on the GPU using CUDA. It demonstrates a basic CUDA kernel and memory management.

## VectorAdditionUnified.cu

This code performs vector addition using Unified Memory, providing a simplified memory management approach for both CPU and GPU.

## MatMul.cu

This CUDA code performs matrix multiplication using a basic kernel. It demonstrates parallelization of matrix operations on the GPU.

## MatMulTiled.cu

This CUDA code optimizes matrix multiplication by using shared memory tiling to enhance memory access patterns and increase performance.

## SumReductionWorkDivergence.cu

This CUDA code demonstrates a simple sum reduction with potential work divergence among threads.

## SumReductionBankConflicts.cu

This CUDA code addresses bank conflicts in sum reduction by modifying memory access patterns.

## SumReductionNoConflicts.cu

This CUDA code optimizes sum reduction by avoiding bank conflicts and improving memory access patterns.

## SumReductionReducedThreads.cu

This CUDA code reduces idle threads in sum reduction by optimizing the number of threads participating in the computation.

## SumReductionDeviceFunction.cu

This CUDA code uses a device function to further optimize sum reduction, removing unnecessary work in the last iteration.

## SumReductionCoopGrp.cu

This CUDA code demonstrates sum reduction using cooperative groups, a feature available in CUDA 9.0 onwards, to improve thread coordination and reduce synchronization overhead.

## Convolution.cu

This CUDA code implements a basic convolution operation on a 1D array. It utilizes parallelism to enhance the performance of the convolution operation. Widely used in machine learning (CNN).

Feel free to explore each code file for more details and optimizations applied.

## ConstMemConvolution.cu

Constant memory in CUDA is a region of memory that is cached on the device and read-only for all threads within a thread block. It is ideal for storing values that do not change during the execution of a kernel. In the provided code, the convolution mask is stored in constant memory. This means that every thread within a thread block can efficiently access the same constant mask without having to fetch it from global memory repeatedly.

**Note:** Ensure you have the necessary dependencies and a compatible GPU before running these CUDA programs. Adjust the compilation and execution commands based on your system configuration.