#version 460 core
#include <flutter/runtime_effect.glsl>

// Input image dimensions
uniform vec2 uInputSize;

// Output dimensions (640x640 for YOLO)
uniform vec2 uOutputSize;

// Input texture
uniform sampler2D uTexture;

// Output color
out vec4 fragColor;

void main() {
  // Get current fragment coordinate
  vec2 fragCoord = FlutterFragCoord().xy;

  // Calculate letterbox transformation
  // Scale to fit while maintaining aspect ratio
  float scale = min(uOutputSize.x / uInputSize.x, uOutputSize.y / uInputSize.y);
  vec2 scaledSize = uInputSize * scale;
  vec2 offset = (uOutputSize - scaledSize) * 0.5;

  // Map output coordinate to input coordinate
  vec2 imageCoord = (fragCoord - offset) / scale;
  vec2 uv = imageCoord / uInputSize;

  // Check if we're in the padding region
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    // Gray padding (114/255 = 0.447)
    fragColor = vec4(114.0/255.0, 114.0/255.0, 114.0/255.0, 1.0);
  } else {
    // Sample from input image and normalize to [0, 1]
    vec4 color = texture(uTexture, uv);
    fragColor = color;
  }
}
