# Test Images

This directory contains test images for ExecuTorch model inference demos.

## Available Images

### Classification Test Images
- **cat.jpg** (37KB) - Cat image for MobileNet classification
- **dog.jpg** (40KB) - Dog image for MobileNet classification
- **car.jpg** (49KB) - Car image for classification testing
- **person.jpg** (41KB) - Person portrait for classification

### Object Detection Test Images
- **street.jpg** (92KB) - Street scene with multiple objects for YOLO detection

## Usage in Code

```dart
// Load a test image from assets
final testImage = await rootBundle.load('assets/images/cat.jpg');

// Or use with image picker replacement
final file = File('assets/images/cat.jpg');
```

## Image Sources

All images are sourced from Unsplash (https://unsplash.com) with appropriate licenses for testing and demonstration purposes.

## Adding New Images

To add new test images:

1. Download image and save to this directory
2. Keep file size reasonable (< 500KB for mobile apps)
3. Use descriptive filenames
4. Update this README
5. Ensure `assets/images/` is in pubspec.yaml assets list
