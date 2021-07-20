#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include <curand_kernel.h>
#pragma warning(push, 0)
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#pragma warning(pop)
#include "ray.h"
#include "hitable.h"

#define M_PI 3.14159265358979323846f

using namespace glm;

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.f - dt * dt);
    if (discriminant > 0.f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 randomUnitVec3(curandState* local_rand_state) {
    if (local_rand_state == nullptr)
        return { 0.0f, 0.0f, 0.0f };
    const float a = 2.0f * M_PI * curand_uniform(local_rand_state);
    const float z = -1.0f + 2.0f * curand_uniform(local_rand_state);
    const float r = sqrtf(1.f - z * z);
    return { r * cos(a), r * sin(a), z };
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, const bool antiAliasing) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, const bool antiAliasing) const {
        vec3 target;
        if (antiAliasing)
        {
            target = rec.p + rec.normal + randomUnitVec3(local_rand_state);
        }
        else
        {
            target = rec.p + rec.normal;
        }
        scattered = ray(rec.p, normalize(target - rec.p));
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1.f) fuzz = f; else fuzz = 1.f; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, const bool antiAliasing) const {
        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        if (antiAliasing)
        {
            scattered = ray(rec.p, reflected + fuzz * randomUnitVec3(local_rand_state));
        }
        else
        {
            scattered = ray(rec.p, reflected);
        }
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, const bool antiAliasing) const {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1.f - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};
#endif