#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

void cudaCaller(cudaError_t command)
{
    cudaError_t const error = command;
    if (cudaSuccess != error) {
        std::cout << cudaGetErrorName(error) << "..." << cudaGetErrorString(error) << std::endl;
        std::cout << cudaGetLastError();
    }
}

#define cudaErrorChk(func) { cudaFuncAssert((func), __FILE__, __LINE__); }
inline void cudaFuncAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cudaFuncAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}

// Choose which GPU to run on, change this on a multi-GPU system.
void setProperCudaDevice(cudaDeviceProp properties, int device, int version)
{
    cudaCaller(cudaGetDevice(&device));
    cudaCaller(cudaDeviceGetAttribute(&version, cudaDevAttrComputeCapabilityMajor, device));
    properties.major = version;
    cudaCaller(cudaChooseDevice(&device, &properties));
    cudaCaller(cudaSetDevice(device));
}

// Calculate right count of threads per block
void setKernelThreadsBlock(int frame_width, int frame_height, dim3& blockSettings, dim3& threadSettings)
{
    int device;
    int maxThreadsPerBlock;
    cudaCaller(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, cudaGetDevice(&device)));

    if (maxThreadsPerBlock >= 8 * 8)
    {
        maxThreadsPerBlock = 8;
    }
    else
    {
        maxThreadsPerBlock = (int)sqrt(maxThreadsPerBlock);
    }
    
    blockSettings.x = frame_width / maxThreadsPerBlock + 1;
    blockSettings.y = frame_height / maxThreadsPerBlock + 1;
    threadSettings.x = maxThreadsPerBlock;
    threadSettings.y = maxThreadsPerBlock;
}

//GPU Info with Error handling
void gpuInfo(cudaDeviceProp properties)
{
    memset(&properties, 0, sizeof(cudaDeviceProp));
    int deviceCounter = 0;
    cudaCaller(cudaGetDeviceCount(&deviceCounter));

    // Cuda device detected
    std::cout << deviceCounter << " CUDA device detected on your computer!" << std::endl << std::endl;

    for (int deviceIndex = 0; deviceIndex < deviceCounter; deviceIndex++)
    {
        cudaCaller(cudaGetDeviceProperties(&properties, deviceIndex));

        // Show Cuda device information
        std::cout << "CUDA device information " << deviceIndex << ":" << std::endl << std::endl;
        std::cout << "Name: " << properties.name << std::endl;
        std::cout << "Version: " << properties.major << "." << properties.minor << std::endl;
        std::cout << "GridSize: x: " << properties.maxGridSize[0] << ", y:" << properties.maxGridSize[1] << ", z:" << properties.maxGridSize[2] << std::endl;
        std::cout << "MaxThreadPerBlock: " << properties.maxThreadsPerBlock << std::endl;
        std::cout << "TotalGlobalMem: " << properties.totalGlobalMem << " Bytes" << std::endl;
        std::cout << "TotalConstMem: " << properties.totalConstMem << " Bytes" << std::endl;
        std::cout << "SharedMemPerBlock: " << properties.sharedMemPerBlock << " Bytes" << std::endl;
        std::cout << "Warp size: " << properties.warpSize << " Threads\n" << std::endl;

        if (cudaGetDevice(&deviceIndex) == 0)
        {
            properties.major = 1;
            properties.minor = 3;
            cudaCaller(cudaChooseDevice(&deviceIndex, &properties));

            cudaCaller(cudaSetDevice(deviceIndex));

            // Cuda device selected
            std::cout << "CUDA device number " << cudaGetDevice(&deviceIndex) << " selected!" << std::endl << std::endl;
        }
    }
}