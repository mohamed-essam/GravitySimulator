#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <algorithm>

using namespace glm;
class Camera
{
protected:
	struct CameraRotation {
		float yaw;
		float pitch;
		float roll;

		float yawChange;
		float pitchChange;
		float rollChange;
		CameraRotation() {
			yaw = yawChange = pitchChange = pitch = roll = rollChange = 0.0f;
		}

	};
	vec3 mUp;
	vec3 mDirection;
	vec3 mRight;
	vec3 mPosition;
	vec3 velocity;

	mat4 mViewMatrix;
	CameraRotation mRotation;
public:
	Camera();

	void virtual Yaw(float angle) = 0;
	void virtual Pitch(float angle) = 0;
	void virtual Roll(float angle) = 0;

	void virtual Walk(float distance) = 0;
	void virtual Strafe(float distance) = 0;
	void virtual Fly(float distance) = 0;

	void virtual UpdateViewMatrix() = 0;
	void virtual Update(float time) = 0;
	void copy(const Camera&);
	void setVelocity(float, float, float);
	void setRotation(float, float, float);
	void setPosition(float, float, float);
	vec3 getVelocity();
	vec3 getPosition();
	CameraRotation getRotation();
	mat4 getViewMatrix();
	vec3 getLookDirection();

	~Camera();
};

