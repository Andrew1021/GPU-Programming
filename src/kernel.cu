#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "kernel.cuh"

void cudaCaller(cudaError_t command) 
{
    cudaError_t const error = command;
    if (cudaSuccess != error) {
        std::cout << cudaGetErrorName(error) << "..." << cudaGetErrorString(error) << std::endl;
    }
}

__global__ void multiplyKernel(int * _gpuReturnMatrix, int* _cudaVector, int* _cudaMatrix)
{
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.x;
    if (threadIndex < N * N)
    {
        _gpuReturnMatrix[Index] = _cudaMatrix[Index] * _cudaVector[threadIndex % N];
    }
}

__global__ void addKernel(int* gpuReturnMatrix, int * _gpuReturnResult)
{
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIndex = threadIdx.x;
    if (threadIndex < N)
    {
        for (int i = 0; i < N; i++)
        {
            _gpuReturnResult[Index] += gpuReturnMatrix[Index * N + i];
        }
    }
}

int multiply(const int* matrix, const int* vector, int * returnArray)
{
    cudaDeviceProp properties;
    properties.major = 5;
    int Device;
    cudaCaller(cudaChooseDevice(&Device, &properties));
    cudaCaller(cudaSetDevice(Device));
    cudaCaller(cudaGetDeviceProperties(&properties, Device));

    int* returnArrayMultiplied = new int[N * N];

    int* gpuReturnMatrix = NULL;
    int* gpuReturnResult = NULL;
    int* gpuArrayMultiplied = NULL;

    int* cudaVector = NULL;
    int* cudaMatrix = NULL;

    const int gpuVectorBufferSize = sizeof(int) * N;
    const int gpuMatrixBufferSize = sizeof(int) * N * N;

    cudaCaller(cudaMalloc((void**)&gpuReturnResult, gpuVectorBufferSize));

    cudaCaller(cudaMalloc((void**)&gpuReturnMatrix, gpuMatrixBufferSize));
    cudaCaller(cudaMalloc((void**)&cudaVector, gpuVectorBufferSize));
    cudaCaller(cudaMalloc((void**)&cudaMatrix, gpuMatrixBufferSize));

    cudaCaller(cudaMemcpy(cudaVector, vector, gpuVectorBufferSize, cudaMemcpyHostToDevice));
    cudaCaller(cudaMemcpy(cudaMatrix, matrix, gpuMatrixBufferSize, cudaMemcpyHostToDevice));
    
    dim3 blockSettings((N * N + properties.maxThreadsPerBlock - 1) / properties.maxThreadsPerBlock);
    dim3 threadSettings(properties.maxThreadsPerBlock);
    multiplyKernel<<<blockSettings, threadSettings>>>(gpuReturnMatrix, cudaVector, cudaMatrix);

    cudaCaller(cudaDeviceSynchronize());

    cudaCaller(cudaMemcpy(returnArrayMultiplied, gpuReturnMatrix, gpuMatrixBufferSize, cudaMemcpyDeviceToHost));

    cudaCaller(cudaMalloc((void**)&gpuArrayMultiplied, gpuMatrixBufferSize));
    cudaCaller(cudaMemcpy(gpuArrayMultiplied, returnArrayMultiplied, gpuMatrixBufferSize, cudaMemcpyHostToDevice));

    dim3 blockSettings2((N + properties.maxThreadsPerBlock - 1) / properties.maxThreadsPerBlock);
    addKernel<<<blockSettings2, threadSettings>>>(gpuArrayMultiplied, gpuReturnResult);

    cudaCaller(cudaMemcpy(returnArray, gpuReturnResult, gpuVectorBufferSize, cudaMemcpyDeviceToHost));

    cudaCaller(cudaFree(gpuReturnResult));
    cudaCaller(cudaFree(gpuReturnMatrix));
    cudaCaller(cudaFree(cudaVector));
    cudaCaller(cudaFree(cudaMatrix));
    cudaCaller(cudaFree(gpuArrayMultiplied));

    return 0;
}