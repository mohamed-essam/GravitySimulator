#include "Camera.h"
Camera::Camera()
{
}
mat4 Camera::getViewMatrix()
{
	return mViewMatrix;
}
vec3 Camera::getLookDirection()
{
	return -1.0f*mDirection;
}
vec3 Camera::getVelocity()
{
	return velocity;
}
vec3 Camera::getPosition()
{
	return mPosition;
}
Camera::CameraRotation Camera::getRotation()
{
	return mRotation;
}
void Camera::setRotation(float p, float y, float r)
{
	mRotation.pitchChange = p;
	mRotation.yawChange = y;
	mRotation.rollChange = r;
}

void Camera::setPosition(float x, float y, float z)
{
	mPosition = vec3(x, y, z);
}

void Camera::copy(const Camera & secondCamera)
{
	this->mDirection = secondCamera.mDirection;
	this->mPosition = secondCamera.mPosition;
	this->mRight = secondCamera.mRight;
	this->mRotation = secondCamera.mRotation;
	this->mUp = secondCamera.mUp;
	this->mViewMatrix = secondCamera.mViewMatrix;
}

void Camera::setVelocity(float x, float y, float z)
{
	velocity = vec3(x, y, z);
}
Camera::~Camera()
{
}
