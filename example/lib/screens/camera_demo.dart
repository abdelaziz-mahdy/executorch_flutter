import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

class CameraDemo extends StatefulWidget {
  const CameraDemo({super.key});

  @override
  State<CameraDemo> createState() => _CameraDemoState();
}

class _CameraDemoState extends State<CameraDemo> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isPermissionGranted = false;
  bool _isProcessing = false;
  String _status = 'Initializing camera...';
  String? _lastResult;

  // Performance metrics
  double _fps = 0.0;
  double _processingTime = 0.0;
  int _frameCount = 0;
  DateTime? _lastFrameTime;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    // Request camera permission
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      setState(() {
        _status = 'Camera permission denied';
      });
      return;
    }

    setState(() {
      _isPermissionGranted = true;
      _status = 'Loading cameras...';
    });

    try {
      // Get available cameras
      _cameras = await availableCameras();
      if (_cameras!.isEmpty) {
        setState(() {
          _status = 'No cameras found';
        });
        return;
      }

      // Initialize camera controller
      _controller = CameraController(
        _cameras!.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();

      setState(() {
        _status = 'Camera ready - tap to start processing';
      });

    } catch (e) {
      setState(() {
        _status = 'Camera initialization failed: $e';
      });
    }
  }

  Future<void> _startImageStream() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    setState(() {
      _isProcessing = true;
      _status = 'Processing camera feed...';
    });

    _controller!.startImageStream((CameraImage image) {
      _processFrame(image);
    });
  }

  Future<void> _stopImageStream() async {
    if (_controller == null) return;

    await _controller!.stopImageStream();
    setState(() {
      _isProcessing = false;
      _status = 'Processing stopped';
    });
  }

  void _processFrame(CameraImage image) {
    if (_isProcessing) {
      _frameCount++;
      final now = DateTime.now();

      if (_lastFrameTime != null) {
        final timeDiff = now.difference(_lastFrameTime!).inMilliseconds;
        if (timeDiff > 0) {
          setState(() {
            _fps = 1000.0 / timeDiff;
          });
        }
      }
      _lastFrameTime = now;

      // Simulate processing time (replace with actual ML inference)
      final processingStart = DateTime.now();
      _simulateProcessing(image).then((result) {
        final processingEnd = DateTime.now();
        final processingTime = processingEnd.difference(processingStart).inMilliseconds.toDouble();

        setState(() {
          _processingTime = processingTime;
          _lastResult = result;
        });
      });
    }
  }

  Future<String> _simulateProcessing(CameraImage image) async {
    // Simulate converting camera frame to tensor format
    await Future.delayed(const Duration(milliseconds: 50));

    // Simulate ML inference
    final results = [
      'Person detected (87%)',
      'Object detected (92%)',
      'Face detected (95%)',
      'Vehicle detected (73%)',
      'Animal detected (81%)',
    ];

    return results[_frameCount % results.length];
  }


  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (!_isPermissionGranted) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Camera Demo'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.camera_alt_outlined,
                size: 64,
                color: theme.colorScheme.onSurface.withOpacity(0.6),
              ),
              const SizedBox(height: 16),
              Text(
                _status,
                style: theme.textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _initializeCamera,
                child: const Text('Request Camera Permission'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Real-time Camera Processing'),
        actions: [
          IconButton(
            icon: Icon(_isProcessing ? Icons.stop : Icons.play_arrow),
            onPressed: _controller?.value.isInitialized == true
                ? (_isProcessing ? _stopImageStream : _startImageStream)
                : null,
          ),
        ],
      ),
      body: _controller?.value.isInitialized == true
          ? Column(
              children: [
                // Camera preview
                Expanded(
                  flex: 3,
                  child: Container(
                    decoration: BoxDecoration(
                      border: Border.all(color: theme.colorScheme.outline),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    margin: const EdgeInsets.all(16),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: CameraPreview(_controller!),
                    ),
                  ),
                ),

                // Status and metrics
                Expanded(
                  flex: 2,
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Status
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(
                                      _isProcessing ? Icons.radio_button_checked : Icons.radio_button_unchecked,
                                      color: _isProcessing ? Colors.green : Colors.grey,
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      'Status',
                                      style: theme.textTheme.titleMedium?.copyWith(
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Text(_status),
                                if (_lastResult != null) ...[
                                  const SizedBox(height: 8),
                                  Container(
                                    padding: const EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                      color: theme.colorScheme.primaryContainer,
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Text(
                                      'Latest: $_lastResult',
                                      style: theme.textTheme.bodyMedium?.copyWith(
                                        color: theme.colorScheme.onPrimaryContainer,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ),

                        const SizedBox(height: 16),

                        // Performance metrics
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(Icons.speed, color: theme.colorScheme.primary),
                                    const SizedBox(width: 8),
                                    Text(
                                      'Performance',
                                      style: theme.textTheme.titleMedium?.copyWith(
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text('FPS', style: theme.textTheme.bodySmall),
                                        Text(
                                          _fps.toStringAsFixed(1),
                                          style: theme.textTheme.titleMedium?.copyWith(
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ],
                                    ),
                                    Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text('Processing', style: theme.textTheme.bodySmall),
                                        Text(
                                          '${_processingTime.toStringAsFixed(1)}ms',
                                          style: theme.textTheme.titleMedium?.copyWith(
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ],
                                    ),
                                    Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text('Frames', style: theme.textTheme.bodySmall),
                                        Text(
                                          _frameCount.toString(),
                                          style: theme.textTheme.titleMedium?.copyWith(
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            )
          : Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_status),
                ],
              ),
            ),
    );
  }
}