#include "cuda_runtime.h"

#define N 50

int multiply(const int* matrix, const int* vector, int* returnArray);
__global__ void multiplyKernel(int* _gpuReturn);
__global__ void addKernel(int* _gpuReturnResult);
void cudaCaller(cudaError_t command);