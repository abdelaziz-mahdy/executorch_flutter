import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Abstract interface for converting CameraImage to JPEG bytes
/// Supports different implementation strategies (image lib, OpenCV)
abstract class CameraImageConverter {
  Future<Uint8List> convertToJpeg(CameraImage cameraImage, {int quality = 85});
}

/// Image library implementation for camera image conversion
/// Uses pure Dart image package for YUV420/BGRA8888 conversion
class ImageLibCameraConverter implements CameraImageConverter {
  @override
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
  }) async {
    img.Image? convertedImage;

    if (Platform.isAndroid) {
      convertedImage = _convertYUV420ToImage(cameraImage);
    } else {
      // iOS and macOS both use BGRA8888
      convertedImage = _convertBGRA8888ToImage(cameraImage);
    }

    if (convertedImage == null) {
      throw Exception('Failed to convert camera image');
    }

    return Uint8List.fromList(img.encodeJpg(convertedImage, quality: quality));
  }

  /// Convert YUV420 (Android) to Image
  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 0;

    final image = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;

        final yp = cameraImage.planes[0].bytes[index];
        final up = cameraImage.planes[1].bytes[uvIndex];
        final vp = cameraImage.planes[2].bytes[uvIndex];

        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }

    return image;
  }

  /// Convert BGRA8888 (iOS) to Image
  img.Image? _convertBGRA8888ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final img.Image image = img.Image(width: width, height: height);
    final Uint8List bytes = cameraImage.planes[0].bytes;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int pixelIndex = (y * width + x) * 4;
        final int b = bytes[pixelIndex];
        final int g = bytes[pixelIndex + 1];
        final int r = bytes[pixelIndex + 2];
        final int a = bytes[pixelIndex + 3];

        image.setPixelRgba(x, y, r, g, b, a);
      }
    }

    return image;
  }
}

/// OpenCV implementation for camera image conversion
/// Uses native OpenCV operations for potentially faster conversion
class OpenCVCameraConverter implements CameraImageConverter {
  @override
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
  }) async {
    final cv.Mat mat;

    if (Platform.isAndroid) {
      mat = await _convertYUV420ToMat(cameraImage);
    } else {
      // iOS and macOS both use BGRA8888
      mat = await _convertBGRA8888ToMat(cameraImage);
    }

    // Convert to RGB (OpenCV uses BGR by default)
    final rgbMat = await cv.cvtColorAsync(mat, cv.COLOR_BGR2RGB);

    // Encode to JPEG using OpenCV
    // imencodeAsync returns (bool success, Uint8List data)
    final (success, encoded) = await cv.imencodeAsync('.jpg', rgbMat);

    // Clean up
    mat.dispose();
    rgbMat.dispose();

    if (!success) {
      throw Exception('Failed to encode camera image to JPEG');
    }

    return encoded;
  }

  /// Convert YUV420 (Android) to OpenCV Mat
  /// Uses native OpenCV YUV420 to BGR conversion
  Future<cv.Mat> _convertYUV420ToMat(CameraImage cameraImage) async {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    // YUV420 format: Y plane (full resolution) + U/V planes (quarter resolution each)
    // We need to create a single-channel Mat with height * 1.5
    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];

    // Combine all planes into a single buffer
    final int ySize = yPlane.bytes.length;
    final int uvSize = uPlane.bytes.length + vPlane.bytes.length;
    final yuvBuffer = Uint8List(ySize + uvSize);

    // Copy Y plane
    yuvBuffer.setRange(0, ySize, yPlane.bytes);

    // Copy U and V planes
    yuvBuffer.setRange(ySize, ySize + uPlane.bytes.length, uPlane.bytes);
    yuvBuffer.setRange(
      ySize + uPlane.bytes.length,
      yuvBuffer.length,
      vPlane.bytes,
    );

    // Create Mat from YUV buffer (single channel, height * 1.5)
    final yuvMat = cv.Mat.create(
      rows: height + height ~/ 2,
      cols: width,
      type: cv.MatType.CV_8UC1,
    );
    yuvMat.data.setAll(0, yuvBuffer);

    // Convert YUV420 to BGR using OpenCV
    // COLOR_YUV2BGR_I420 is for planar YUV420 (I420/IYUV format)
    final bgrMat = await cv.cvtColorAsync(yuvMat, cv.COLOR_YUV2BGR_I420);

    yuvMat.dispose();
    return bgrMat;
  }

  /// Convert BGRA8888 (iOS) to OpenCV Mat
  /// Uses native OpenCV BGRA to BGR conversion
  Future<cv.Mat> _convertBGRA8888ToMat(CameraImage cameraImage) async {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final Uint8List bytes = cameraImage.planes[0].bytes;

    // Create BGRA Mat (4 channels)
    final bgraMat = cv.Mat.create(
      rows: height,
      cols: width,
      type: cv.MatType.CV_8UC4,
    );
    bgraMat.data.setAll(0, bytes);

    // Convert BGRA to BGR (remove alpha channel)
    final bgrMat = await cv.cvtColorAsync(bgraMat, cv.COLOR_BGRA2BGR);

    bgraMat.dispose();
    return bgrMat;
  }
}
