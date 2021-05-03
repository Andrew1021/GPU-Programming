#include "kernel.cuh"
#include <iostream>
#include <stdlib.h>

int main()
{
    int matrix[N * N];
    int vector[N];
    int returnArray[N] = { 0 };

    for (int i = 0; i < N; i++)
    {
        vector[i] = rand() % 10 + 1;
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = rand() % 10 + 1;
            std::cout << matrix[i * N + j] << ", ";
        }
    }
    std::cout <<  "vector ";
    for (int i = 0; i < N; i++)
    {
        std::cout << vector[i] << ", ";
    }

    multiply(matrix, vector, returnArray);

    std::cout << "Result: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << returnArray[i] << ", ";
    }

	return 0;
}