
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void kernel( void )
{
    // Do something
}

int main()
{
    kernel<<<1, 1 >>>();

    // Hello World
    std::cout << "Hello world!" << std::endl;

    // Uebung 1 - Abfrage GPU Eigenschaften
    cudaDeviceProp properties;
    memset(& properties, 0, sizeof( cudaDeviceProp ));
    int deviceCounter = 0;
    cudaGetDeviceCount( &deviceCounter );
    for ( int deviceIndex = 0; deviceIndex < deviceCounter; deviceIndex++ )
    {
        cudaGetDeviceProperties( &properties, deviceIndex );

        std::cout << "Name: " << properties.name << std::endl;
        std::cout << "Version: " << properties.major << "." << properties.minor << std::endl;
        std::cout << "GridSize: x: " << properties.maxGridSize[0] << ", y:" << properties.maxGridSize[1] << ", z:" << properties.maxGridSize[2] << std::endl;
        std::cout << "MaxThreadPerBlock: " << properties.maxThreadsPerBlock << std::endl;
        std::cout << "TotalGlobalMem: " << properties.totalGlobalMem << " Bytes" << std::endl;
        std::cout << "TotalConstMem: " << properties.totalConstMem << " Bytes" << std::endl;
        std::cout << "SharedMemPerBlock: " << properties.sharedMemPerBlock << " Bytes" << std::endl;
        std::cout << "Warp size: " << properties.warpSize << " Threads" << std::endl;
    }
    
    properties.major = 1;
    properties.minor = 3;
    int deviceIndex = 0;
    cudaChooseDevice( &deviceIndex, &properties );

    cudaSetDevice( deviceIndex );

    return 0;
}
