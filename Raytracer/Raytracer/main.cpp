/*
* A CUDA powered 3D real-time ray tracer.
* Windowing and graphics done with GLFW and OpenGL.
*
*/

#include <iostream>
#include <stdlib.h>
#include <glew.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <sstream>
#include <string>

#pragma warning(push, 0)
#include <glm/vec3.hpp>
#pragma warning(pop)

#include "ray.h"
#include "kernel.cuh"

using namespace glm;

const float distanceBetweenPixels = 1.f;
const float distanceToPixels = 100.f;
__constant__ ray* rays = new ray[WIDTH * HEIGHT];
__constant__ vec3* colors = new vec3[WIDTH * HEIGHT];
float milliseconds = 0;

const std::string WINDOW_TITLE_BASE = "GPU CUDA RayTracer Cubes";

bool quit_program = false;

unsigned long long int num_frames = 0;
double time_delta = 0.0;


// OpenGL/Cuda rendering
GLuint gl_texturePtr;
GLuint gl_pixelBufferObject;
cudaGraphicsResource* cgr;
uchar4* d_textureBufData;

// Key flags
bool w_key_pressed = false;
bool a_key_pressed = false;
bool s_key_pressed = false;
bool d_key_pressed = false;
bool e_key_pressed = false;
bool q_key_pressed = false;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto key_action_assignment = [&action](bool& flag) {
        if (action == GLFW_PRESS)
            flag = true;
        else if (action == GLFW_RELEASE)
            flag = false;
    };

    switch (key)
    {
    case GLFW_KEY_ESCAPE: key_action_assignment(quit_program);          break;
    case GLFW_KEY_W: key_action_assignment(w_key_pressed);              break;
    case GLFW_KEY_A: key_action_assignment(a_key_pressed);              break;
    case GLFW_KEY_S: key_action_assignment(s_key_pressed);              break;
    case GLFW_KEY_D: key_action_assignment(d_key_pressed);              break;
    case GLFW_KEY_E: key_action_assignment(e_key_pressed);              break;
    case GLFW_KEY_Q: key_action_assignment(q_key_pressed);              break;
    default: break;
    }
}

void moveLight()
{
    if (w_key_pressed)
        moveForward();
    if (a_key_pressed)
        moveLeft();
    if (s_key_pressed)
        moveBackward();
    if (d_key_pressed)
        moveRight();
    if (q_key_pressed)
        moveUp();
    if (e_key_pressed)
        moveDown();
}

void UpdateFPS(GLFWwindow* window)
{
    static const unsigned int WINDOW_UPDATE_INTERVAL = 4;


    num_frames++;
    time_delta += milliseconds / 1000.f;

    if (time_delta > 1.0 / WINDOW_UPDATE_INTERVAL)
    {
        double FPS = num_frames / time_delta;
        num_frames = 0;
        time_delta = 0;

        std::stringstream title;

        title << WINDOW_TITLE_BASE << " FPS: " << FPS;
        glfwSetWindowTitle(window, title.str().c_str());
    }
}

void initBuffers()
{
    void* data = malloc((HEIGHT * WIDTH * COLOR_DEPTH) * sizeof(GLubyte));

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_texturePtr);
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glGenBuffers(1, &gl_pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (size_t)HEIGHT * WIDTH * sizeof(uchar4), data, GL_DYNAMIC_DRAW);

    free(data);
    cudaGraphicsGLRegisterBuffer(&cgr, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void renderScene()
{
    CUDA_CALLER(cudaGraphicsMapResources(1, &cgr, 0));
    size_t num_bytes;
    CUDA_CALLER(cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufData, &num_bytes, cgr));

    // Cuda function
    render(rays, distanceBetweenPixels, distanceToPixels, colors, d_textureBufData, milliseconds);

    CUDA_CALLER(cudaGraphicsUnmapResources(1, &cgr, 0));

    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(float(WIDTH), 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(float(WIDTH), float(HEIGHT));
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, float(HEIGHT));
    glEnd();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void createPPM()
{
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";

    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            int index = j * WIDTH + i;
            int ir = static_cast<int>(255.999 * colors[index].r);
            int ig = static_cast<int>(255.999 * colors[index].g);
            int ib = static_cast<int>(255.999 * colors[index].b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}

int main()
{
    // Initialize the GLFW library
    if (!glfwInit())
        return -1;

    // Create a windowed mode window and its OpenGL context 
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_TITLE_BASE.data(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Set openGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Make the window's context current
    glfwMakeContextCurrent(window);
    glewInit();

    glDisable(GL_DEPTH_TEST);

    // Assign callbacks to window 
    glfwSetKeyCallback(window, key_callback);

    int buf_w, buf_h;
    glfwGetFramebufferSize(window, &buf_w, &buf_h);
    glViewport(0, 0, buf_w, buf_h);
    glOrtho(0.0f, WIDTH, HEIGHT, 0.0f, 0.0f, 1.0f);

    // Cuda function
    setupWorld(rays, distanceToPixels, distanceBetweenPixels);

    initBuffers();

    // Loop until the user closes the window 
    while (!glfwWindowShouldClose(window) && !quit_program)
    {
        // Render here 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderScene();

        // Swap front and back buffers 
        glfwSwapBuffers(window);

        // Poll for and process events 
        glfwPollEvents();
        moveLight();

        UpdateFPS(window);
    }
    //createPPM();
    glDeleteTextures(1, &gl_texturePtr);
    glDeleteBuffers(1, &gl_pixelBufferObject);
    cleanUp();
    glfwTerminate();

    return 0;
}
