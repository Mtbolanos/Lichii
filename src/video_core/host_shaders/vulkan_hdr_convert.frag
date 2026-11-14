// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Converts a presented layer between SDR, HDR10 and scRGB. Based on Eglantine #557's
// hdr_tonemap.frag (PQ to SDR) and hdr_pq_encode.frag (SDR to PQ) by 101Leafy, with the
// luminances the game was told instead of fixed ones, and scRGB on either side.

#version 460

layout (location = 0) in vec4 posPos;

layout (location = 0) out vec4 frag_color;

layout (binding = 1) uniform sampler2D input_texture;

// 0: sRGB BT.709 (SDR), 1: PQ BT.2020 (HDR10), 2: linear BT.709 (scRGB).
layout (push_constant) uniform Params {
    uint input_encoding;
    uint output_encoding;
    float reference_white; // nits SDR white is shown at
    float peak;            // nits an SDR output rolls highlights off to
    float nits_per_unit;   // nits of linear 1.0 in a scRGB input
    uint premultiplied;    // colour is premultiplied by alpha
};

const float PQ_M1 = 0.1593017578125;
const float PQ_M2 = 78.84375;
const float PQ_C1 = 0.8359375;
const float PQ_C2 = 18.8515625;
const float PQ_C3 = 18.6875;

const float SCRGB_NITS = 80.0;

const mat3 BT2020_TO_BT709 = mat3(
    1.6605, -0.1246, -0.0182,
   -0.5876,  1.1329, -0.1006,
   -0.0728, -0.0083,  1.1187);

const mat3 BT709_TO_BT2020 = mat3(
    0.6274, 0.0691, 0.0164,
    0.3293, 0.9195, 0.0880,
    0.0433, 0.0114, 0.8956);

vec3 PqToNits(vec3 e) {
    vec3 p = pow(clamp(e, 0.0, 1.0), vec3(1.0 / PQ_M2));
    return 10000.0 * pow(max(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), vec3(1.0 / PQ_M1));
}

vec3 NitsToPq(vec3 nits) {
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(PQ_M1));
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), vec3(PQ_M2));
}

vec3 SrgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 LinearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Linear BT.709 in nits.
vec3 Decode(vec3 c) {
    if (input_encoding == 1) {
        return BT2020_TO_BT709 * PqToNits(c);
    }
    if (input_encoding == 2) {
        return c * nits_per_unit;
    }
    return SrgbToLinear(clamp(c, 0.0, 1.0)) * reference_white;
}

vec3 Encode(vec3 nits) {
    if (output_encoding == 1) {
        return NitsToPq(max(BT709_TO_BT2020 * nits, 0.0));
    }
    if (output_encoding == 2) {
        return nits / SCRGB_NITS;
    }
    vec3 c = max(nits / reference_white, 0.0);
    // Extended Reinhard on the brightest channel, so hue holds as highlights roll off
    float m = max(max(c.r, c.g), c.b);
    if (m > 0.0) {
        float w = max(peak / reference_white, 1.0);
        c *= (1.0 + m / (w * w)) / (1.0 + m);
    }
    return LinearToSrgb(clamp(c, 0.0, 1.0));
}

void main() {
    vec4 src = texture(input_texture, posPos.xy);
    if (premultiplied != 0) {
        vec3 color = src.a > 0.0 ? src.rgb / src.a : vec3(0.0);
        frag_color = vec4(Encode(Decode(color)) * src.a, src.a);
        return;
    }
    frag_color = vec4(Encode(Decode(src.rgb)), src.a);
}
