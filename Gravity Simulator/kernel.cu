#pragma region Includes
#include <gl/glew.h>
#include <gl/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "shader.hpp"
#include "FPCamera.h"
#pragma endregion

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define EPS 1e-1

#pragma region Variables

const long long COUNT = 1000;
__device__ const float SPEED = 0.1f;
const float G_CONSTANT = 0.01f;
const float POINT_SIZE = 3.0f;
const int MAX_MASS = 250;
const int MIN_MASS = 100;
const float TERMINAL_VELOCITY = 15.0f;
__device__ const float MOUSE_MASS = 2000.0f;

float mTime;
GLuint positionsVBO;
GLuint VAO;

struct cudaGraphicsResource* positionsVBO_CUDA;

float3* velocity;
float3* forces;
float* masses;

GLfloat* positions;
GLFWwindow* mWindow;
double mouseX=0, mouseY=0;

#pragma endregion

void Update(float time);
void Draw();

#pragma region Kernels
__global__ void simulateFrame(float* positions, float3* velocity, float delta, float particleRadius, float3* forces, float* particleMass, float terminalVelocity)
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

__global__ void calculateForce(float* positions, float3* forces, float* particleMass, float gravityConstant, int idx) {
	long long ii = threadIdx.x + blockIdx.x * blockDim.x;
	int id = ii;
	if (id >= COUNT || id == idx) {
		return;
	}
	float x1, x2, y1, y2, z1, z2;
	x1 = positions[idx * 6];
	y1 = positions[idx * 6 + 1];
	z1 = positions[idx * 6 + 2];
	x2 = positions[id * 6];
	y2 = positions[id * 6 + 1];
	z2 = positions[id * 6 + 2];
	float distance = ((x1 - x2)*(x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)) + EPS;
	float force = gravityConstant * particleMass[idx] * particleMass[id] / distance;
	float3 vec = make_float3(x1 - x2, y1 - y2, z1 - z2);
	float mag = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z) + EPS;
	vec.x /= mag, vec.y /= mag, vec.z /= mag;
	forces[id].x += vec.x * force;
	forces[id].y += vec.y * force;
	forces[id].z += vec.z * force;
}

__global__ void calculateForceMouse(float* positions, float3* forces, float* particleMass, float gravityConstant, float mouseX, float mouseY) {
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

__global__ void setVal(float3* arr, float3 val) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= COUNT) return;
	arr[idx] = val;
}
#pragma endregion

GLuint MVP_ID;
glm::mat4 projectionMatrix;
FPCamera* cam;
int lastKey = -1;

void handleKeyPress(GLFWwindow*, int, int, int, int);

int main(int argv, char ** argc)
{
	float3* h_velocity = new float3[COUNT];

	srand(time(NULL));

	for (int idx = 0; idx < COUNT; idx++)
	{
		h_velocity[idx].x = rand() % 10 + 1;
		if (rand() % 2)
			h_velocity[idx].x *= -1;
		h_velocity[idx].y = rand() % 10 + 1;
		if (rand() % 2)
			h_velocity[idx].y *= -1;
		h_velocity[idx].z = rand() % 10 + 1;
		if (rand() % 2)
			h_velocity[idx].z *= -1;
	}

	float* h_masses = new float[COUNT];

	for (int i = 0; i < COUNT; i++) {
		h_masses[i] = rand() % (MAX_MASS - MIN_MASS) + MIN_MASS;
	}


	cudaMalloc(&masses, COUNT * sizeof(float));
	cudaMemcpy(masses, h_masses, COUNT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&velocity, COUNT * sizeof(float3));
	cudaMemcpy(velocity, h_velocity, COUNT * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc(&forces, COUNT*sizeof(float3));
	setVal <<<(COUNT + 1023) / 1024, 1024 >>>(forces, make_float3(0, 0, 0));

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
	int screenWidth = mode->width;
	int screenHeight = mode->height;

	mWindow = glfwCreateWindow(screenWidth, screenHeight, "Gravity Sim", monitor, NULL);

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
		positions[i * 6] = (((float)(rand())) / RAND_MAX) * 10 - 5;
		positions[i * 6 + 1] = (((float)(rand())) / RAND_MAX) * 10 - 5;
		positions[i * 6 + 2] = (((float)(rand())) / RAND_MAX) * 10 - 5;
		positions[i * 6 + 3] = 1 - MAX(0, h_masses[i] / MAX_MASS - 0.5f) * 2;
		positions[i * 6 + 4] = 0;
		positions[i * 6 + 5] = MAX(0, h_masses[i] / MAX_MASS - 0.5f) * 2;
	}

	glfwSetKeyCallback(mWindow, &handleKeyPress);

	MVP_ID = glGetUniformLocation(programID, "mvp");
	projectionMatrix = glm::perspective(45.0f, (float)(mode->width) / mode->height, 0.5f, 100.0f);
	cam = new FPCamera();

	glBufferData(GL_ARRAY_BUFFER, COUNT * 6 * sizeof(GLfloat), positions, GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	setVal <<<(COUNT+1023)/1024,1024 >>>(forces, make_float3(0, 0, 0));
	for (int i = 0; i < COUNT; i++)
	{
		calculateForce <<<(COUNT+1023)/1024, 1024 >>>(positions, forces, masses, G_CONSTANT, i);
	}
	glfwGetCursorPos(mWindow, &mouseX, &mouseY);
	glfwSetCursorPos(mWindow, 1920.0f / 2, 1080.0f / 2);
	//calculateForceMouse <<<(COUNT+1023)/1024, 1024 >>> (positions, forces, masses, G_CONSTANT, mouseX/700.0f*2.0f-1.0f, (700-mouseY) / 700.0f * 2.0f - 1.0f);
	simulateFrame <<<(COUNT + 1023) / 1024, 1024 >>>(positions, velocity, time, POINT_SIZE, forces, masses, TERMINAL_VELOCITY);
	if (lastKey == GLFW_KEY_W) {
		cam->Walk(time * 5);
	}
	if (lastKey == GLFW_KEY_S) {
		cam->Walk(time * -5);
	}
	if (lastKey == GLFW_KEY_A) {
		cam->Strafe(time * 5);
	}
	if (lastKey == GLFW_KEY_D) {
		cam->Strafe(time * -5);
	}
	cam->Update(0);
	cam->Yaw(-mouseX + 1920.0f / 2);
	cam->Pitch(mouseY - 1080.0f / 2);
}

void Draw() {
	cam->UpdateViewMatrix();
	glm::mat4 viewMatrix = cam->getViewMatrix();
	glm::mat4 modelMatrix(1.0f);
	glm::mat4 MVP = projectionMatrix * viewMatrix * modelMatrix;

	glUniformMatrix4fv(MVP_ID, 1, GL_FALSE, &MVP[0][0]);
	
	glDrawArrays(GL_POINTS, 0, COUNT);
}

void handleKeyPress(GLFWwindow * window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_RELEASE) {
		lastKey = key;
	}
	else
		lastKey = -1;
}
