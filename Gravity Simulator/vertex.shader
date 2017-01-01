#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;

out vec3 color;

uniform mat4 mvp;
uniform vec3 cam_loc;

void main() {
	gl_Position = mvp * vec4(pos,1);
	color = col;
	float dist = length(cam_loc - gl_Position.xyz);
	float att = sqrt(1.0 / (1 +
		(3 +
			7 * dist) * dist));
	float size = clamp(300 * att, 1, 10);
	gl_PointSize = size;
}
