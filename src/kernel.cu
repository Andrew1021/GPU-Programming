#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "kernel.cuh"

__constant__ int cudaVector[N];
__constant__ int cudaMatrix[N * N];
__constant__ int cudaMatrixMultiplied[N * N];

void cudaCaller(cudaError_t command) 
{
    cudaError_t const error = command;
    if (cudaSuccess != error) {
        std::cout << cudaGetErrorName(error) << "..." << cudaGetErrorString(error) << std::endl;
    }
}

__global__ void multiplyKernel(int * _gpuReturn)
{
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.x;
    if (threadIndex < N * N)
    {
        _gpuReturn[Index] = cudaMatrix[Index] * cudaVector[threadIndex % N];
    }
}

__global__ void addKernel(int * _gpuReturnResult)
{
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.x;
    if (threadIndex < N)
    {
        for (int i = 0; i < N; i++)
        {
            _gpuReturnResult[Index] += cudaMatrixMultiplied[Index * N + i];
        }
    }
}

int multiply(const int* matrix, const int* vector, int * returnArray)
{
    cudaDeviceProp properties;
    cudaCaller(cudaGetDeviceProperties(&properties, 0));

    int* gpuReturnMatrix = NULL;
    int* gpuReturnResult = NULL;

    const int gpuVectorBufferSize = sizeof(int) * N;
    const int gpuMatrixBufferSize = sizeof(int) * N * N;

    cudaCaller(cudaMalloc((void**)&gpuReturnMatrix, gpuMatrixBufferSize));
    cudaCaller(cudaMalloc((void**)&gpuReturnResult, gpuVectorBufferSize));

    cudaCaller(cudaMemcpyToSymbol(cudaMatrix, matrix, gpuMatrixBufferSize));
    cudaCaller(cudaMemcpyToSymbol(cudaVector, vector, gpuVectorBufferSize));

    dim3 blockSettings((N * N + properties.maxThreadsPerBlock - 1) / properties.maxThreadsPerBlock);
    dim3 threadSettings(properties.maxThreadsPerBlock);
    multiplyKernel<<<blockSettings, threadSettings>>>(gpuReturnMatrix);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaCaller(cudaMemcpyToSymbol(cudaMatrixMultiplied, gpuReturnMatrix, gpuMatrixBufferSize));

    dim3 blockSettings2((N + properties.maxThreadsPerBlock - 1) / properties.maxThreadsPerBlock);
    addKernel<<<blockSettings2, threadSettings>>>(gpuReturnResult);

    cudaCaller(cudaMemcpy(returnArray, gpuReturnResult, gpuVectorBufferSize, cudaMemcpyDeviceToHost));

    cudaCaller(cudaFree(gpuReturnMatrix));

    return 0;
}