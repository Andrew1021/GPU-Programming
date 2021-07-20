#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#pragma warning(push, 0)
#include <glm/vec3.hpp>
#pragma warning(pop)

#include "hitable.h"
#include "diffuse_light.h"
#include "ray.h"

#define WIDTH 640
#define HEIGHT 360

#define COLOR_DEPTH 4

#define SINGLE_FLT_MAX 3.402823466e+20f        // max value
#define RND (curand_uniform(&local_rand_state))

#define CUDA_CALLER( code ) \
    { \
        cudaError_t const error = code; \
        if (cudaSuccess != error) \
        { \
            std::cerr << cudaGetErrorName(error) << " ... " << cudaGetErrorString(error) << " ... Errorcode: " << cudaGetLastError() << std::endl; \
        }   \
    } \

using namespace glm;

__global__ void setupRaysKernel(ray* rays, const float distanceBetweenPixels, const float distanceToPixels, curandState* rand_state);
__global__ void renderKernel(ray* rays, vec3* colors, hitable** world, curandState* rand_state, diffuse_light** light, uchar4* d_textureBufData, const float distanceToPixels);
__global__ void create_world(hitable** d_list, hitable** d_world, diffuse_light** d_light, curandState* local_rand_state);

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, diffuse_light** light, const bool antiAliasing);
__device__ vec3 randomUnitVec3(curandState* local_rand_state);

void render(ray* rays, const float distanceBetweenPixels, const float distanceToPixels, vec3* colors, uchar4* d_textureBufData, float& milliseconds);
void setupWorld(ray* rays, const float distanceToPixels, const float distanceBetweenPixels);
void cleanUp();

__global__ void moveForwardKernel(diffuse_light** light);
void moveForward();

__global__ void moveBackwardKernel(diffuse_light** light);
void moveBackward();

__global__ void moveLeftKernel(diffuse_light** light);
void moveLeft();

__global__ void moveRightKernel(diffuse_light** light);
void moveRight();

__global__ void moveUpKernel(diffuse_light** light);
void moveUp();

__global__ void moveDownKernel(diffuse_light** light);
void moveDown();

