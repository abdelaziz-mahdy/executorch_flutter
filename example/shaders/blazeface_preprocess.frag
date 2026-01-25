#version 460 core
#include <flutter/runtime_effect.glsl>

// Input image dimensions
uniform vec2 uInputSize;

// Output dimensions (128x128 for BlazeFace)
uniform vec2 uOutputSize;

// Input texture
uniform sampler2D uTexture;

// Output color
out vec4 fragColor;

void main() {
  // Get current fragment coordinate
  vec2 fragCoord = FlutterFragCoord().xy;

  // Simple resize (no letterbox, no aspect ratio preservation)
  // BlazeFace expects direct resize to 128x128
  vec2 uv = fragCoord / uOutputSize;

  // Check if we're outside the valid region (shouldn't happen with simple resize)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    // Black padding for areas outside input (normalized to [-1, 1])
    fragColor = vec4(-1.0, -1.0, -1.0, 1.0);
  } else {
    // Sample from input image
    vec4 color = texture(uTexture, uv);

    // Normalize to [-1, 1] range: (pixel / 127.5) - 1.0
    // Since texture returns [0, 1], we convert: (value * 255 / 127.5) - 1.0 = value * 2.0 - 1.0
    vec3 normalized = color.rgb * 2.0 - 1.0;

    fragColor = vec4(normalized, 1.0);
  }
}
