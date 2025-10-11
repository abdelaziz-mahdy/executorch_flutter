import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Abstract interface for converting CameraImage to JPEG bytes
/// Supports different implementation strategies (image lib, OpenCV)
abstract class CameraImageConverter {
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
    int? sensorOrientation,
  });
}

/// Image library implementation for camera image conversion
/// Uses pure Dart image package for YUV420/BGRA8888 conversion
class ImageLibCameraConverter implements CameraImageConverter {
  @override
  Future<Uint8List> convertToJpeg(
    CameraImage cameraImage, {
    int quality = 85,
    int? sensorOrientation,
  }) async {
    img.Image? convertedImage;

    if (Platform.isAndroid) {
      convertedImage = _convertYUV420ToImage(cameraImage);
      // Apply rotation based on sensor orientation
      if (convertedImage != null && sensorOrientation != null && sensorOrientation != 0) {
        convertedImage = img.copyRotate(convertedImage, angle: sensorOrientation);
      }
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

    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];

    final image = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yPlane.bytesPerRow + x;
        final int uvRow = (y / 2).floor();
        final int uvCol = (x / 2).floor();
        final int uvIndex = uvRow * uPlane.bytesPerRow + uvCol * uPlane.bytesPerPixel!;

        final int yValue = yPlane.bytes[yIndex];
        final int uValue = uPlane.bytes[uvIndex];
        final int vValue = vPlane.bytes[uvIndex];

        // YUV to RGB conversion
        int r = (yValue + 1.370705 * (vValue - 128)).round().clamp(0, 255);
        int g = (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
            .round()
            .clamp(0, 255);
        int b = (yValue + 1.732446 * (uValue - 128)).round().clamp(0, 255);

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
    int? sensorOrientation,
  }) async {
    cv.Mat mat;

    if (Platform.isAndroid) {
      mat = await _convertYUV420ToMat(cameraImage);
      // Apply rotation based on sensor orientation
      if (sensorOrientation != null && sensorOrientation != 0) {
        final rotateCode = _getRotateCode(sensorOrientation);
        if (rotateCode != null) {
          final rotatedMat = cv.rotate(mat, rotateCode);
          mat.dispose();
          mat = rotatedMat;
        }
      }
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

  /// Convert sensor orientation to OpenCV rotate code
  int? _getRotateCode(int sensorOrientation) {
    switch (sensorOrientation) {
      case 90:
        return cv.ROTATE_90_CLOCKWISE;
      case 180:
        return cv.ROTATE_180;
      case 270:
        return cv.ROTATE_90_COUNTERCLOCKWISE;
      default:
        return null;
    }
  }

  /// Convert YUV420 (Android) to OpenCV Mat
  /// Uses native OpenCV YUV420 to BGR conversion
  Future<cv.Mat> _convertYUV420ToMat(CameraImage cameraImage) async {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];

    // Check if UV planes are interleaved (NV21/NV12) or planar (I420)
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    if (uvPixelStride == 2) {
      // NV21 format (semi-planar, interleaved VU)
      // Create Y plane mat
      final yMat = cv.Mat.create(
        rows: height,
        cols: width,
        type: cv.MatType.CV_8UC1,
      );
      yMat.data.setAll(0, yPlane.bytes);

      // Create UV plane mat (interleaved)
      final uvHeight = height ~/ 2;
      final uvWidth = width ~/ 2;
      final uvMat = cv.Mat.create(
        rows: uvHeight,
        cols: uvWidth,
        type: cv.MatType.CV_8UC2,
      );
      uvMat.data.setAll(0, uPlane.bytes);

      // Combine Y and UV into YUV Mat
      final yuvMat = cv.Mat.create(
        rows: height + height ~/ 2,
        cols: width,
        type: cv.MatType.CV_8UC1,
      );
      yuvMat.data.setRange(0, yPlane.bytes.length, yPlane.bytes);
      yuvMat.data.setRange(yPlane.bytes.length, yuvMat.data.length, uPlane.bytes);

      // Convert NV21 to BGR
      final bgrMat = await cv.cvtColorAsync(yuvMat, cv.COLOR_YUV2BGR_NV21);

      yMat.dispose();
      uvMat.dispose();
      yuvMat.dispose();
      return bgrMat;
    } else {
      // I420 format (planar)
      final int ySize = yPlane.bytes.length;
      final int uvSize = uPlane.bytes.length + vPlane.bytes.length;
      final yuvBuffer = Uint8List(ySize + uvSize);

      yuvBuffer.setRange(0, ySize, yPlane.bytes);
      yuvBuffer.setRange(ySize, ySize + uPlane.bytes.length, uPlane.bytes);
      yuvBuffer.setRange(
        ySize + uPlane.bytes.length,
        yuvBuffer.length,
        vPlane.bytes,
      );

      final yuvMat = cv.Mat.create(
        rows: height + height ~/ 2,
        cols: width,
        type: cv.MatType.CV_8UC1,
      );
      yuvMat.data.setAll(0, yuvBuffer);

      final bgrMat = await cv.cvtColorAsync(yuvMat, cv.COLOR_YUV2BGR_I420);

      yuvMat.dispose();
      return bgrMat;
    }
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
