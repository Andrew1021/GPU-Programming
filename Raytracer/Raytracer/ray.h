#pragma once

#ifndef RAYH
#define RAYH
#pragma warning(push, 0)
#include <glm/vec3.hpp>
#pragma warning(pop)

#include "cuda_runtime.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const glm::vec3& origin, const glm::vec3& direction) { 
        _origin = origin; 
        _direction = direction;
    }
    __device__ glm::vec3 origin() const { return _origin; }
    __device__ glm::vec3 direction() const { return _direction; }
    __device__ glm::vec3 point_at_parameter(float t) const { return _origin + t * _direction; }

    glm::vec3 _origin;
    glm::vec3 _direction;
};

#endif