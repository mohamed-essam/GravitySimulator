#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Camera.h"

class FPCamera : public Camera
{
public:
	FPCamera();

	void Move(glm::vec3 dis);
	void Yaw(float angle);
	void Pitch(float angle);
	void Roll(float angle);

	void Walk(float distance);
	void Strafe(float distance);
	void Fly(float distance);

	void UpdateViewMatrix();
	void Update(float time);

	~FPCamera();
};

