#version 460 core
#include <flutter/runtime_effect.glsl>

// Input image dimensions
uniform vec2 uInputSize;

// Output dimensions (192x192 for Lightning, 256x256 for Thunder)
uniform vec2 uOutputSize;

// Input texture
uniform sampler2D uTexture;

// Output color
out vec4 fragColor;

void main() {
  // Get current fragment coordinate
  vec2 fragCoord = FlutterFragCoord().xy;

  // Simple resize (no letterbox, no aspect ratio preservation)
  // MoveNet expects direct resize to target size
  vec2 uv = fragCoord / uOutputSize;

  // Check if we're outside the valid region (shouldn't happen with simple resize)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    // Black padding for areas outside input
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    // Sample from input image
    vec4 color = texture(uTexture, uv);

    // Normalize to [0, 1] range (texture already returns [0, 1])
    // MoveNet expects RGB values in [0, 1] range
    fragColor = color;
  }
}
