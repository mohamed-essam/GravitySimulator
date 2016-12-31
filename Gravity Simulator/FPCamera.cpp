#include "FPCamera.h"

FPCamera::FPCamera()
{
	mPosition = vec3(0);
	mDirection = vec3(0, 0, -1);
	mUp = vec3(0, 1, 0);
	mRight = vec3(1, 0, 0);
	UpdateViewMatrix();
}


void FPCamera::Move(glm::vec3 dis) {
	glm::vec3 newPos = dis + mPosition;
	mPosition = newPos;
}

void FPCamera::Walk(float distance) {
	Move(-mDirection * distance);
}

void FPCamera::Strafe(float distance) {
	Move(mRight * distance);
}

void FPCamera::Fly(float distance) {
	Move(mUp * distance);
}

void FPCamera::Roll(float angle) {
	mRotation.roll += angle;
	mat4 rotationMat = rotate(mRotation.roll, mDirection);
	vec4 tmp = rotationMat * vec4(mUp, 1.0f);
	mUp = vec3(tmp.x, tmp.y, tmp.z);
	tmp = rotationMat * vec4(mRight, 1.0f);
	mRight = vec3(tmp.x, tmp.y, tmp.z);
}

void FPCamera::Pitch(float angle) {
	mRotation.pitch += angle;
	mRotation.pitch = std::max(-90.0f, std::min(90.0f, mRotation.pitch));
	mat4 rotationMat = rotate(mRotation.pitch, mRight);
	vec4 tmp = rotationMat * vec4(mUp, 1.0f);
	mUp = vec3(tmp.x, tmp.y, tmp.z);
	tmp = rotationMat * vec4(mDirection, 1.0f);
	mDirection = vec3(tmp.x, tmp.y, tmp.z);
}

void FPCamera::Yaw(float angle) {
	mRotation.yaw += angle;
	mat4 rotationMat = rotate(mRotation.yaw, mUp);
	vec4 tmp = rotationMat * vec4(mDirection, 1.0f);
	mDirection = vec3(tmp.x, tmp.y, tmp.z);
	tmp = rotationMat * vec4(mRight, 1.0f);
	mRight = vec3(tmp.x, tmp.y, tmp.z);
}

void FPCamera::UpdateViewMatrix() {
	vec3 center = mPosition + this->getLookDirection();
	mViewMatrix = lookAt(mPosition, center, mUp);
}

void FPCamera::Update(float time) {
	mUp = vec3(0, 1, 0);
	mRight = vec3(1, 0, 0);
	mDirection = vec3(0, 0, -1);
	Yaw(mRotation.yawChange*time);
	Pitch(mRotation.pitchChange*time);
	Roll(mRotation.rollChange*time);
	Walk(velocity.z * time);
	Strafe(velocity.x * time);
}

FPCamera::~FPCamera()
{
}