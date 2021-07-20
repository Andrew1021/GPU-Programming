#pragma once

#pragma warning(push, 0)
#include <glm/vec3.hpp>
#pragma warning(pop)

#include "ray.h"

using namespace glm;

class diffuse_light
{
public:
	vec3 origin = { 0.0f, 0.0f, 0.0f };  // origin of the diffuse light at first point
	int width = 1, height = 5;             // width and height of the area
	int numberOfLightPoints = 10;
	float intensity = 0.3f;			       // intensity of each light point
	vec3* positionOfLightPoints = new vec3[numberOfLightPoints * numberOfLightPoints];
	vec3 normal = vec3(0.f, -1.f, 0.f);
	float step = 5.f;

	__device__ diffuse_light(vec3 _origin, int _width, int _height, int _numberOfLightPoints, float _intensity) : origin{ _origin }, width{ _width }, height{ _height }, numberOfLightPoints{ _numberOfLightPoints }, intensity{ _intensity }{
		float intensityPerLightPoint = 1.f / (float)numberOfLightPoints;
		intensity *= intensityPerLightPoint;
		positionOfLightPoints = new vec3[numberOfLightPoints * numberOfLightPoints];
		updateLightPoints();

	}

	__device__ void updateLightPoints()
	{
		for (int i = 0; i < numberOfLightPoints; i++)
		{
			float offsetZ = i * (float)((float)height / (float)numberOfLightPoints);
			for (int j = 0; j < numberOfLightPoints; j++)
			{
				float offsetX = j * (float)((float)width / (float)numberOfLightPoints);
				positionOfLightPoints[i + numberOfLightPoints * j] = vec3(origin.x + offsetX, origin.y, origin.z + offsetZ);
			}
		}
	}

	__device__ void moveForward()
	{
		origin = vec3(origin.x, origin.y, origin.z + step);
		updateLightPoints();
	}
	__device__ void moveBackward()
	{
		origin = vec3(origin.x, origin.y, origin.z - step);
		updateLightPoints();
	}

	__device__ void moveLeft()
	{
		origin = vec3(origin.x - step, origin.y, origin.z);
		updateLightPoints();
	}
	__device__ void moveRight()
	{
		origin = vec3(origin.x + step, origin.y, origin.z);
		updateLightPoints();
	}

	__device__ void moveUp() {
		origin = vec3(origin.x, origin.y + step, origin.z);
		updateLightPoints();
	}

	__device__ void moveDown()
	{
		origin = vec3(origin.x, origin.y - step, origin.z);
		updateLightPoints();
	}

};