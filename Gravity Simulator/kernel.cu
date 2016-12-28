#pragma region Includes
#include <gl/glew.h>
#include <gl/glfw3.h>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "shader.hpp"
#pragma endregion

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define EPS 2

#pragma region Variables

const long long COUNT = 2000;
__device__ const float SPEED = 0.1f;
const float G_CONSTANT = 0.1f;
const float POINT_SIZE = 3.0f;
const float TERMINAL_VELOCITY = 25.0f;
__device__ const float MOUSE_MASS = 250.0f;

float mTime;
GLuint positionsVBO;
GLuint VAO;

struct cudaGraphicsResource* positionsVBO_CUDA;

float2* velocity;
float2* forces;
float* masses;

GLfloat* positions;
GLFWwindow* mWindow;
double mouseX=0, mouseY=0;

#pragma endregion

void Update(float time);
void Draw();

#pragma region Kernels
__global__ void simulateFrame(float* positions, float2* velocity, float delta, float particleRadius, float2* forces, float* particleMass, float terminalVelocity)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= COUNT) return;
	delta *= SPEED;
	velocity[idx].x += delta * forces[idx].x / particleMass[idx];
	velocity[idx].y += delta * forces[idx].y / particleMass[idx];
	velocity[idx].x = MIN(terminalVelocity, velocity[idx].x);
	velocity[idx].y = MIN(terminalVelocity, velocity[idx].y);
	velocity[idx].x = MAX(-terminalVelocity, velocity[idx].x);
	velocity[idx].y = MAX(-terminalVelocity, velocity[idx].y);
	positions[idx * 6] += velocity[idx].x * delta;
	positions[idx * 6 + 1] += velocity[idx].y * delta;
	if (positions[idx * 6] < -1 ) {
		velocity[idx].x *= -1;
		positions[idx * 6] = -1;
	}
	if (positions[idx * 6] > 1) {
		velocity[idx].x *= -1;
		positions[idx * 6] = 1;
	}
	if (positions[idx * 6 + 1] < -1) {
		velocity[idx].y *= -1;
		positions[idx * 6 + 1] = -1;
	}
	if (positions[idx * 6 + 1] > 1) {
		velocity[idx].y *= -1;
		positions[idx * 6 + 1] = 1;
	}
}

__global__ void calculateForce(float* positions, float2* forces, float* particleMass, float gravityConstant, int idx) {
	long long ii = threadIdx.x + blockIdx.x * blockDim.x;
	int id = ii;
	if (id >= COUNT || id == idx) {
		return;
	}
	float x1, x2, y1, y2;
	x1 = positions[idx * 6];
	y1 = positions[idx * 6 + 1];
	x2 = positions[id * 6];
	y2 = positions[id * 6 + 1];
	float distance = ((x1 - x2)*(x1 - x2) + (y1 - y2) * (y1 - y2)) + EPS;
	float force = gravityConstant * particleMass[idx] * particleMass[id] / distance;
	float2 vec = make_float2(x1 - x2, y1 - y2);
	float mag = sqrt(vec.x*vec.x + vec.y*vec.y) + EPS;
	vec.x /= mag, vec.y /= mag;
	forces[id].x += vec.x * force;
	forces[id].y += vec.y * force;
}

__global__ void calculateForceMouse(float* positions, float2* forces, float* particleMass, float gravityConstant, float mouseX, float mouseY) {
	long long ii = threadIdx.x + blockIdx.x * blockDim.x;
	int id = ii;
	if (id >= COUNT) {
		return;
	}
	float x1, x2, y1, y2;
	x1 = mouseX;
	y1 = mouseY;
	x2 = positions[id * 6];
	y2 = positions[id * 6 + 1];
	float distance = ((x1 - x2)*(x1 - x2) + (y1 - y2) * (y1 - y2)) + EPS;
	float force = gravityConstant * MOUSE_MASS * particleMass[id] / distance;
	float2 vec = make_float2(x1 - x2, y1 - y2);
	float mag = sqrt(vec.x*vec.x + vec.y*vec.y) + EPS;
	vec.x /= mag, vec.y /= mag;
	forces[id].x += vec.x * force;
	forces[id].y += vec.y * force;
}

__global__ void setVal(float2* arr, float2 val) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= COUNT) return;
	arr[idx] = val;
}
#pragma endregion

int main(int argv, char ** argc)
{
	float2* h_velocity = new float2[COUNT];

	//srand(time(NULL));

	for (int idx = 0; idx < COUNT; idx++)
	{
		h_velocity[idx].x = rand() % 10 + 1;
		if (rand() % 2)
			h_velocity[idx].x *= -1;
		h_velocity[idx].y = rand() % 10 + 1;
		if (rand() % 2)
			h_velocity[idx].y *= -1;
	}

	float* h_masses = new float[COUNT];

	for (int i = 0; i < COUNT; i++) {
		h_masses[i] = rand() % 150 + 100;
	}


	cudaMalloc(&masses, COUNT * sizeof(float));
	cudaMemcpy(masses, h_masses, COUNT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&velocity, COUNT * sizeof(float2));
	cudaMemcpy(velocity, h_velocity, COUNT * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMalloc(&forces, COUNT*sizeof(float2));
	setVal <<<(COUNT + 1023) / 1024, 1024 >>>(forces, make_float2(0, 0));

	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, false);

	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	int screenWidth = 700;
	int screenHeight = 700;

	mWindow = glfwCreateWindow(screenWidth, screenHeight, "Gravity Sim", NULL, NULL);

	if (mWindow == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(mWindow);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return;
	}
	glfwSetInputMode(mWindow, GLFW_STICKY_KEYS, GL_TRUE);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * (sizeof(GLfloat)), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * (sizeof(GLfloat)), (void*)(3 * sizeof(GLfloat)));

	GLuint programID = LoadShaders("vertex.shader", "fragment.shader");
	glUseProgram(programID);

	mTime = glfwGetTime();

	glPointSize(POINT_SIZE);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	positions = new GLfloat[COUNT * 6];
	for (int i = 0; i < COUNT; i++) {
		positions[i * 6] = (((float)(rand())) / RAND_MAX) * 2 - 1;
		positions[i * 6 + 1] = (((float)(rand())) / RAND_MAX) * 2 - 1;
		positions[i * 6 + 2] = 0;
		positions[i * 6 + 3] = 1 - MAX(0, h_masses[i] / 250.0f - 0.5f) * 2;
		positions[i * 6 + 4] = 0;
		positions[i * 6 + 5] = MAX(0, h_masses[i] / 250.0f - 0.5f) * 2;
	}


	glBufferData(GL_ARRAY_BUFFER, COUNT * 6 * sizeof(GLfloat), positions, GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);

	do {
		float time = (float)(glfwGetTime() - mTime);
		mTime = glfwGetTime();
		Update(time);
		Draw();
		glfwSwapBuffers(mWindow);
		glfwPollEvents();
	} while (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(mWindow) == 0);
    return 0;
}

void Update(float time) {
	glClear(GL_COLOR_BUFFER_BIT);
	setVal <<<(COUNT+1023)/1024,1024 >>>(forces, make_float2(0, 0));
	for (int i = 0; i < COUNT; i++)
	{
		calculateForce <<<(COUNT+1023)/1024, 1024 >>>(positions, forces, masses, G_CONSTANT, i);
	}
	glfwGetCursorPos(mWindow, &mouseX, &mouseY);
	calculateForceMouse <<<(COUNT+1023)/1024, 1024 >>> (positions, forces, masses, G_CONSTANT, mouseX/500.0f*2.0f-1.0f, (500-mouseY) / 500.0f * 2.0f - 1.0f);
	simulateFrame <<<(COUNT + 1023) / 1024, 1024 >>>(positions, velocity, time, POINT_SIZE, forces, masses, TERMINAL_VELOCITY);
}

void Draw() {
	glDrawArrays(GL_POINTS, 0, COUNT);
}