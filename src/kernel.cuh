#include "cuda_runtime.h"

#define N 30

int multiply(const int* matrix, const int* vector, int* returnArray);
__global__ void multiplyKernel(int* _gpuReturnMatrix, int* _cudaVector, int* _cudaMatrix);
__global__ void addKernel(int* gpuReturnMatrix, int* _gpuReturnResult);
void cudaCaller(cudaError_t command);