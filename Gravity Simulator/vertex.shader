#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;

out vec3 color;

void main() {
	gl_Position.xyz = pos;
	gl_Position.w = 1;
	color = col;
}
