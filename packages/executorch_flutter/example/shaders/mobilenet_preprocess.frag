#version 460 core
#include <flutter/runtime_effect.glsl>

// Input image dimensions
uniform vec2 uInputSize;

// Output dimensions (224x224 for MobileNet)
uniform vec2 uOutputSize;

// Input texture
uniform sampler2D uTexture;

// ImageNet normalization constants
// mean = [0.485, 0.456, 0.406]
// std = [0.229, 0.224, 0.225]
const vec3 mean = vec3(0.485, 0.456, 0.406);
const vec3 std = vec3(0.229, 0.224, 0.225);

// Output color
out vec4 fragColor;

void main() {
  // Get current fragment coordinate
  vec2 fragCoord = FlutterFragCoord().xy;

  // Calculate center crop transformation
  // First, resize to maintain aspect ratio with shortest side = 256
  float scale = max(256.0 / uInputSize.x, 256.0 / uInputSize.y);
  vec2 scaledSize = uInputSize * scale;

  // Center crop to 224x224
  vec2 cropOffset = (scaledSize - uOutputSize) * 0.5;

  // Map output coordinate to input coordinate
  vec2 scaledCoord = fragCoord + cropOffset;
  vec2 inputCoord = scaledCoord / scale;
  vec2 uv = inputCoord / uInputSize;

  // Check if we're outside the valid region
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    // Black padding for areas outside input
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    // Sample from input image
    vec4 color = texture(uTexture, uv);

    // Normalize using ImageNet mean and std
    // (pixel - mean) / std
    vec3 normalized = (color.rgb - mean) / std;

    fragColor = vec4(normalized, 1.0);
  }
}
