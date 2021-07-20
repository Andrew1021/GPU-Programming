#pragma once

#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

using namespace glm;

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    vec3 color;
    material* mat_ptr;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif