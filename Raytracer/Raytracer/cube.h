#pragma once

#ifndef CUBEH
#define CUBEH

#include "cuda_runtime.h"
#include <math.h>

#pragma warning(push, 0)
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#pragma warning(pop)

#include "ray.h"
#include "hitable.h"


using namespace glm;

class cube: public hitable
{
public:
	__device__ cube(vec3 _center, float _diameter, vec3 _color, material* m) : center(_center), diameter(_diameter), cube_color(_color), mat_ptr(m) { }
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    vec3 normal = normalize(cross((vec3(diameter, 0.f, 0.f) - b0), (vec3(0.f, diameter, 0.f) - b0)));
    vec3 center;
    float diameter;
    vec3 cube_color;
    material* mat_ptr;
    bool reflective;
    vec3 radius = vec3((diameter / 2.f));
    vec3 b0 = { vec3(center.x - diameter / 2.f, center.y - diameter / 2.f, center.z - diameter / 2.f) };
    vec3 b1 = { vec3(center.x + diameter / 2.f, center.y + diameter / 2.f, center.z + diameter / 2.f) };
};

// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
__device__ bool cube::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t;
    for (int i = 0; i < 3; ++i) {
        const float invD = 1.0f / r.direction()[i];
        float t0 = (b0[i] - r.origin()[i]) * invD;
        float t1 = (b1[i] - r.origin()[i]) * invD;
        if (invD < 0.0f) {
            const float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;

        t = t_min;

        if (t < 0.f) {
            t = t_max;
            if (t < 0.f) return false;
        }
    }

    // normal: https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-2/blob/master/tutorial2_cuda_pathtracer.cu
    vec3 normal = vec3(0.f, 0.f, 0.f);
    float epsilon = 0.001f; // required to prevent self intersection

    vec3 point = r.point_at_parameter(t);
    if (fabs(b0.x - point.x) < epsilon) normal = vec3(-1.f, 0.f, 0.f);
    else if (fabs(b1.x - point.x) < epsilon) normal = vec3(1.f, 0.f, 0.f);
    else if (fabs(b0.y - point.y) < epsilon) normal = vec3(0.f, -1.f, 0.f);
    else if (fabs(b1.y - point.y) < epsilon) normal = vec3(0.f, 1.f, 0.f);
    else if (fabs(b0.z - point.z) < epsilon) normal = vec3(0.f, 0.f, -1.f);
    else normal = vec3(0.f, 0.f, 1.f);

    normal = normalize(normal);
    normal = dot(normal, r.direction()) < 0.f ? normal : normal * vec3(-1.f);  // correctly oriented normal
    rec.t = t;

    rec.normal = normal;

    rec.color = cube_color;

    rec.p = point;

    rec.mat_ptr = mat_ptr;

    return true;
}

#endif