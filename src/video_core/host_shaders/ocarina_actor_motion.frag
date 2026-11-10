#version 450
#define OCARINA_COMPARE_SCENE_DEPTH 1

layout(location = 0) in vec4 previous_clip;
layout(location = 0) out vec2 motion;

#ifdef OCARINA_COMPARE_SCENE_DEPTH
layout(binding = 3) uniform sampler2D scene_depth;
layout(push_constant) uniform DepthComparison {
    uint depth_tolerance;
} comparison;
#endif

layout(std140, binding = 2, row_major) uniform Transforms {
    mat4 current_view_projection;
    mat4 previous_view_projection;
    vec4 current_origin;
    vec4 previous_origin;
    vec2 extent;
    vec2 jitter;
} transforms;

void main() {
#ifdef OCARINA_COMPARE_SCENE_DEPTH
    uint expected = floatBitsToUint(texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r);
    uint actual = floatBitsToUint(gl_FragCoord.z);
    if (max(expected, actual) - min(expected, actual) > comparison.depth_tolerance) {
        discard;
    }
#endif
    if (previous_clip.w <= 1e-6) {
        discard;
    }
    vec2 previous_uv = previous_clip.xy / previous_clip.w * vec2(0.5, -0.5) + 0.5;
    // The draw is offset by the scene's jitter so it covers the same samples as the game's draw;
    // the motion goes from the unjittered position, as FSR expects motion without jitter.
    vec2 displacement = previous_uv * transforms.extent - (gl_FragCoord.xy - transforms.jitter);
    if (!all(lessThanEqual(abs(displacement), vec2(65504)))) {
        discard;
    }
    motion = displacement;
}
