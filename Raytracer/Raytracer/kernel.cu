#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

#pragma warning(push, 0)
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>
#pragma warning(pop)

#include "cube.h"
#include "kernel.cuh"
#include "hitable_list.h"
#include "diffuse_light.h"
#include "material.h"
#include "ray.h"
#include "cuda_utils.h"

using namespace glm;

__constant__ ray* gpuRays = NULL;
__constant__ vec3* gpuColor = NULL;

__constant__ hitable** d_list;
__constant__ hitable** d_world;
__constant__ diffuse_light** d_light;
__constant__ curandState* d_rand_state;

const int gpuRaysSize = sizeof(ray) * WIDTH * HEIGHT;
const int gpuColorSize = sizeof(vec3) * WIDTH * HEIGHT;
const int numberOfSamples = 100;
const int rayDepth = 3;
const int numberOfCubes = 4;
const int numberOfLightPoints = 3;
const bool antiAliasing = true;

__global__ void moveForwardKernel(diffuse_light** light)
{
    light[0]->moveForward();
}

void moveForward()
{
    moveForwardKernel << <1, 1 >> > (d_light);
}

__global__ void moveBackwardKernel(diffuse_light** light)
{
    light[0]->moveBackward();
}

void moveBackward()
{
    moveBackwardKernel << <1, 1 >> > (d_light);
}

__global__ void moveLeftKernel(diffuse_light** light)
{
    light[0]->moveLeft();
}

void moveLeft()
{
    moveLeftKernel << <1, 1 >> > (d_light);
}

__global__ void moveRightKernel(diffuse_light** light)
{
    light[0]->moveRight();
}

void moveRight()
{
    moveRightKernel << <1, 1 >> > (d_light);
}

__global__ void moveUpKernel(diffuse_light** light)
{
    light[0]->moveUp();
}

void moveUp()
{
    moveUpKernel << <1, 1 >> > (d_light);
}

__global__ void moveDownKernel(diffuse_light** light)
{
    light[0]->moveDown();
}

void moveDown()
{
    moveDownKernel << <1, 1 >> > (d_light);
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, diffuse_light** light, const bool antiAliasing) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.f);
    float light_attenuation = 0.2f;
    ray scattered;
    vec3 attenuation, unit_direction, c;
    float distanceToLightSource, angleToLightSource, t;

    for (int i = 0; i < rayDepth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.0001f, SINGLE_FLT_MAX, rec)) {
            // material
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state, antiAliasing)) {
                cur_ray = scattered;
                cur_attenuation *= attenuation;
            }
            // light
            for (int j = 0; j < light[0]->numberOfLightPoints; j++)
            {
                for (int k = 0; k < light[0]->numberOfLightPoints; k++)
                {
                    hit_record recLight;
                    ray rayToLightSource;
                    if (antiAliasing)
                    {
                        rayToLightSource = ray(rec.p, normalize(light[0]->positionOfLightPoints[j + light[0]->numberOfLightPoints * k] - rec.p + 0.2f * randomUnitVec3(local_rand_state)));
                    }
                    else
                    {
                        rayToLightSource = ray(rec.p, normalize(light[0]->positionOfLightPoints[j + light[0]->numberOfLightPoints * k] - rec.p));
                    }
                    if (!(*world)->hit(rayToLightSource, 0.0001f, SINGLE_FLT_MAX, recLight))
                    {
                        distanceToLightSource = length(normalize(rayToLightSource.direction()));

                        angleToLightSource = angle(normalize(rayToLightSource.direction()), normalize(light[0]->normal));

                        light_attenuation += light[0]->intensity * angleToLightSource * distanceToLightSource;

                        if (light_attenuation > 1.f)
                        {
                            light_attenuation = 1.f;
                        }
                    }
                }
            }
        }
        // No object in world was hit (sky)
        else {
            // If sky was hit with first ray light shouldnt effect the color
            if (i == 0)
            {
                unit_direction = normalize(cur_ray.direction());
                t = 0.5f * (unit_direction.y + 1.0f);
                return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            }
            unit_direction = normalize(cur_ray.direction());
            t = 0.5f * (unit_direction.y + 1.0f);
            c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return c * cur_attenuation * light_attenuation;
        }
    }
    return cur_attenuation * light_attenuation;
}

__global__ void renderKernel(ray* rays, vec3* colors, hitable** world, curandState* rand_state, diffuse_light** light, uchar4* d_textureBufData, const float distanceToPixels)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= WIDTH) || (j >= HEIGHT)) return;
    const int index = j * WIDTH + i;
    curandState local_rand_state = rand_state[index];
    vec3 col = { 0.f, 0.f, 0.f };
    float x, y;

    // Anti-aliasing 
    if (antiAliasing)
    {
        for (int s = 0; s < numberOfSamples; s++) {
            x = rays[index].direction().x + 500.f * float(curand_uniform(&local_rand_state)) / float(WIDTH);
            y = rays[index].direction().y - 500.f * float(curand_uniform(&local_rand_state)) / float(HEIGHT);
            col += color(ray(vec3(0.f, 0.f, 0.f), vec3(x, y, distanceToPixels)), world, &local_rand_state, light, antiAliasing);
        }
        rand_state[index] = local_rand_state;
        col /= float(numberOfSamples);
    }
    else
    {
        col = color(ray(vec3(0.f, 0.f, 0.f), vec3(rays[index].direction().x, rays[index].direction().y, distanceToPixels)), world, &local_rand_state, light, antiAliasing);
        rand_state[index] = local_rand_state;
    }
    
    // Gamma correction
    colors[index] = vec3(sqrt(col.x), sqrt(col.y), sqrt(col.z));

    d_textureBufData[index].x = 255.f * colors[index].r;
    d_textureBufData[index].y = 255.f * colors[index].g;
    d_textureBufData[index].z = 255.f * colors[index].b;
    d_textureBufData[index].w = 255.f;
}

__global__ void create_world(hitable** d_list, hitable** d_world, diffuse_light** d_light, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        // Floor cube
        *(d_list) = new cube(vec3(0.f, -10.5f, 9.f), 15.f, vec3(0.5333f, 0.5333f, 0.5333f), new metal(vec3(0.5333f, 0.5333f, 0.5333f), 0.05f));


        // Random cubes
        int i = 1;
        for (int a = -2; a < numberOfCubes - 2; a++) {
            for (int b = 0; b < numberOfCubes; b++) {
                d_list[i++] = new cube(vec3(a * 3.f + RND, -2.5f, b * 3 + 3.f + RND - 0.5f), 0.5f + RND, vec3(RND, RND, RND), new lambertian(vec3(RND, RND, RND)));
            }
        }

        // Fix scene
        /**(d_list + 1) = new cube(vec3(3.f, -1.9f, 3.f), 2.f, vec3(0.f, 0.f, 1.f), new lambertian(vec3(0.f, 0.f, 1.f)), false);
        *(d_list + 2) = new cube(vec3(0.f, -2.4f, 2.5f), 1.f, vec3(0.f, 1.f, 0.f), new lambertian(vec3(0.f, 1.f, 0.f)), false);
        *(d_list + 3) = new cube(vec3(-3.f, -2.4f, 3.f), 1.f, vec3(1.f, 0.f, 0.f), new lambertian(vec3(1.f, 0.f, 0.f)), false);*/
        
        *rand_state = local_rand_state;

        *d_world = new hitable_list(d_list, numberOfCubes * numberOfCubes + 1);

        *(d_light) = new diffuse_light(vec3(45.f, 5.f, -25.f), 3, 3, numberOfLightPoints, 0.3f);
    }
}

__global__ void setupRaysKernel(ray* rays, const float distanceBetweenPixels, const float distanceToPixels, curandState* rand_state) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= WIDTH) || (j >= HEIGHT)) return;
    const int pixel_index = j * WIDTH + i;
    vec3 direction;
    direction = { -(WIDTH / 2.f * distanceBetweenPixels) + i * distanceBetweenPixels + 0.5f, HEIGHT / 2.f * distanceBetweenPixels - j * distanceBetweenPixels - 0.5f, distanceToPixels };
    rays[pixel_index] = ray(vec3(0.f, 0.f, 0.f), direction);
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(clock() + pixel_index, 0.f, 0.f, &rand_state[pixel_index]);
}

void setupWorld(ray* rays, const float distanceToPixels, const float distanceBetweenPixels)
{
    cudaDeviceProp properties;
    properties.major = 5;
    int Device;
    CUDA_CALLER(cudaChooseDevice(&Device, &properties));
    CUDA_CALLER(cudaSetDevice(Device));
    CUDA_CALLER(cudaGetDeviceProperties(&properties, Device));

    CUDA_CALLER(cudaMalloc((void**)&gpuRays, gpuRaysSize));

    CUDA_CALLER(cudaMalloc((void**)&gpuColor, gpuColorSize));

    const int num_pixels = WIDTH * HEIGHT;
    // allocate random state
    CUDA_CALLER(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    dim3 blockSettings, threadSettings;
    setKernelThreadsBlock(WIDTH, HEIGHT, blockSettings, threadSettings);

    // Output console
    std::cerr << "Rendering a " << WIDTH << "x" << HEIGHT << " image with " << numberOfSamples << " samples per pixel and " << rayDepth << " ray depth ";
    std::cerr << "in " << blockSettings.x << "x" << blockSettings.y << " blocks and " << threadSettings.x << "x" << threadSettings.y << " threads.\n";

    setupRaysKernel<<<blockSettings, threadSettings>>>(gpuRays, distanceBetweenPixels, distanceToPixels, d_rand_state);

    CUDA_CALLER(cudaGetLastError());
    CUDA_CALLER(cudaDeviceSynchronize());

    CUDA_CALLER(cudaMalloc((void**)&d_list, (numberOfCubes * numberOfCubes + 1) * sizeof(hitable*)));

    CUDA_CALLER(cudaMalloc((void**)&d_world, sizeof(hitable*)));

    CUDA_CALLER(cudaMalloc((void**)&d_light, sizeof(diffuse_light*)));

    create_world << <1, 1 >> > (d_list, d_world, d_light, d_rand_state);

    CUDA_CALLER(cudaGetLastError());
    CUDA_CALLER(cudaDeviceSynchronize());
}

void render(ray* rays, const float distanceBetweenPixels, const float distanceToPixels, vec3* colors, uchar4* d_textureBufData, float& milliseconds)
{   
    cudaEvent_t cuda_start, cuda_stop;
    CUDA_CALLER(cudaEventCreate(&cuda_start));
    CUDA_CALLER(cudaEventCreate(&cuda_stop));
    CUDA_CALLER(cudaEventRecord(cuda_start));

    dim3 blockSettings, threadSettings;
    setKernelThreadsBlock(WIDTH, HEIGHT, blockSettings, threadSettings);
    renderKernel<<<blockSettings, threadSettings>>>(gpuRays, gpuColor, d_world, d_rand_state, d_light, d_textureBufData, distanceToPixels);
    
    CUDA_CALLER(cudaGetLastError());
    CUDA_CALLER(cudaDeviceSynchronize());
    CUDA_CALLER(cudaEventRecord(cuda_stop));
    CUDA_CALLER(cudaEventSynchronize(cuda_stop));
    CUDA_CALLER(cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop));

   // std::cerr << "It took " << milliseconds << " milliseconds to render the image with cuda.\n";

    CUDA_CALLER(cudaMemcpy(rays, gpuRays, gpuRaysSize, cudaMemcpyDeviceToHost));
    CUDA_CALLER(cudaMemcpy(colors, gpuColor, gpuColorSize, cudaMemcpyDeviceToHost));
}

void cleanUp()
{
    CUDA_CALLER(cudaFree(gpuRays));
    CUDA_CALLER(cudaFree(gpuColor));
    CUDA_CALLER(cudaFree(d_rand_state));
    CUDA_CALLER(cudaFree(d_list));
    CUDA_CALLER(cudaFree(d_world));
    CUDA_CALLER(cudaFree(d_light));
}