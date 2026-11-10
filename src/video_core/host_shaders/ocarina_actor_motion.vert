#version 450
#define OCARINA_EIGHT_WEIGHTS 1

layout(location = 0) in vec3 position;
#ifdef OCARINA_EIGHT_WEIGHTS
layout(location = 4) in vec4 weights;
layout(location = 5) in vec4 extra_weights;
layout(location = 6) in uvec4 joints;
layout(location = 7) in uvec4 extra_joints;
#else
layout(location = 4) in vec3 weights;
layout(location = 6) in uvec3 joints;
#endif
layout(location = 0) out vec4 previous_clip;

layout(std430, binding = 0, row_major) readonly buffer CurrentPose {
    mat4x3 bones[];
} current_pose;
layout(std430, binding = 1, row_major) readonly buffer PreviousPose {
    mat4x3 bones[];
} previous_pose;
layout(std140, binding = 2, row_major) uniform Transforms {
    mat4 current_view_projection;
    mat4 previous_view_projection;
    vec4 current_origin;
    vec4 previous_origin;
    vec2 extent;
    vec2 jitter;
} transforms;

void blend_joint(vec4 local, uint joint, float weight, inout vec3 current, inout vec3 previous) {
    if (weight != 0 && joint < uint(current_pose.bones.length()) &&
        joint < uint(previous_pose.bones.length())) {
        current += (current_pose.bones[joint] * local) * weight;
        previous += (previous_pose.bones[joint] * local) * weight;
    }
}

void main() {
    vec4 local = vec4(position, 1);
    vec3 current = vec3(0);
    vec3 previous = vec3(0);
    for (int i = 0; i < weights.length(); ++i) {
        blend_joint(local, joints[i], weights[i], current, previous);
    }
#ifdef OCARINA_EIGHT_WEIGHTS
    for (int i = 0; i < 4; ++i) {
        blend_joint(local, extra_joints[i], extra_weights[i], current, previous);
    }
#endif
    gl_Position = transforms.current_view_projection * vec4(current + transforms.current_origin.xyz, 1);
    previous_clip = transforms.previous_view_projection * vec4(previous + transforms.previous_origin.xyz, 1);
    // Match Zelda's [-1, 1] clip depth before Vulkan rasterization.
    gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;
}
