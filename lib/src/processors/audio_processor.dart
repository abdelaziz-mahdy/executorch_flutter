import 'dart:math' as math;
import 'dart:typed_data';

import 'package:meta/meta.dart';

import '../generated/executorch_api.dart';
import 'base_processor.dart';

/// Configuration for audio preprocessing
@immutable
class AudioPreprocessConfig {
  const AudioPreprocessConfig({
    this.sampleRate = 16000,
    this.windowSize = 1024,
    this.hopLength = 512,
    this.nMels = 80,
    this.nFFT = 1024,
    this.normalizeAudio = true,
    this.applyMelSpectrogram = false,
  });

  /// Target sample rate for audio (default: 16kHz)
  final int sampleRate;

  /// Window size for audio processing
  final int windowSize;

  /// Hop length for windowing
  final int hopLength;

  /// Number of mel bands for mel spectrogram
  final int nMels;

  /// FFT size for spectrogram computation
  final int nFFT;

  /// Whether to normalize audio amplitude
  final bool normalizeAudio;

  /// Whether to convert to mel spectrogram
  final bool applyMelSpectrogram;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AudioPreprocessConfig &&
          runtimeType == other.runtimeType &&
          sampleRate == other.sampleRate &&
          windowSize == other.windowSize &&
          hopLength == other.hopLength &&
          nMels == other.nMels &&
          nFFT == other.nFFT &&
          normalizeAudio == other.normalizeAudio &&
          applyMelSpectrogram == other.applyMelSpectrogram;

  @override
  int get hashCode =>
      sampleRate.hashCode ^
      windowSize.hashCode ^
      hopLength.hashCode ^
      nMels.hashCode ^
      nFFT.hashCode ^
      normalizeAudio.hashCode ^
      applyMelSpectrogram.hashCode;
}

/// Result of audio classification
@immutable
class AudioClassificationResult {
  const AudioClassificationResult({
    required this.className,
    required this.confidence,
    required this.classIndex,
    required this.allProbabilities,
    this.audioDurationMs,
  });

  /// The predicted class name/label
  final String className;

  /// Confidence score for the prediction (0.0 to 1.0)
  final double confidence;

  /// Index of the predicted class
  final int classIndex;

  /// All class probabilities (softmax outputs)
  final List<double> allProbabilities;

  /// Duration of analyzed audio in milliseconds (optional)
  final int? audioDurationMs;

  @override
  String toString() =>
      'AudioClassificationResult(class: $className, confidence: ${(confidence * 100).toStringAsFixed(1)}%, duration: ${audioDurationMs}ms)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AudioClassificationResult &&
          runtimeType == other.runtimeType &&
          className == other.className &&
          confidence == other.confidence &&
          classIndex == other.classIndex;

  @override
  int get hashCode =>
      className.hashCode ^ confidence.hashCode ^ classIndex.hashCode;
}

/// Preprocessor for audio data to tensor conversion
class AudioClassificationPreprocessor extends ExecuTorchPreprocessor<Float32List> {
  AudioClassificationPreprocessor({
    required this.config,
  });

  final AudioPreprocessConfig config;

  @override
  String get inputTypeName => 'Audio (Float32List)';

  @override
  bool validateInput(Float32List input) {
    return input.isNotEmpty && input.length >= config.windowSize;
  }

  @override
  Future<List<TensorData>> preprocess(Float32List input, {ModelMetadata? metadata}) async {
    try {
      // Process audio data
      var processedAudio = input;

      // Normalize audio if requested
      if (config.normalizeAudio) {
        processedAudio = _normalizeAudio(processedAudio);
      }

      // Resample if needed (simplified - in practice use proper resampling)
      // This is a placeholder for actual resampling logic

      late Float32List features;
      if (config.applyMelSpectrogram) {
        // Convert to mel spectrogram
        features = _computeMelSpectrogram(processedAudio);
      } else {
        // Use raw audio or apply windowing
        features = _applyWindowing(processedAudio);
      }

      // Create tensor based on feature type
      late List<int> shape;
      if (config.applyMelSpectrogram) {
        // Mel spectrogram shape: [batch, time_frames, mel_bins]
        final timeFrames = ((processedAudio.length - config.nFFT) / config.hopLength).floor() + 1;
        shape = [1, timeFrames, config.nMels];
      } else {
        // Raw audio or windowed audio shape: [batch, samples] or [batch, windows, window_size]
        shape = [1, features.length];
      }

      final tensor = ProcessorTensorUtils.createTensor(
        shape: shape,
        dataType: TensorType.float32,
        data: features,
        name: 'audio_input',
      );

      return [tensor];
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PreprocessingException('Audio preprocessing failed: $e', e);
    }
  }

  Float32List _normalizeAudio(Float32List audio) {
    // Find max absolute value
    double maxAbs = 0.0;
    for (final sample in audio) {
      maxAbs = math.max(maxAbs, sample.abs());
    }

    if (maxAbs == 0.0) return audio;

    // Normalize to [-1, 1] range
    return Float32List.fromList(
      audio.map((sample) => sample / maxAbs).toList()
    );
  }

  Float32List _applyWindowing(Float32List audio) {
    // Apply simple windowing (overlapping windows)
    final windows = <double>[];
    final windowFunction = _generateHannWindow(config.windowSize);

    for (int start = 0; start + config.windowSize <= audio.length; start += config.hopLength) {
      for (int i = 0; i < config.windowSize; i++) {
        final sample = audio[start + i] * windowFunction[i];
        windows.add(sample);
      }
    }

    return Float32List.fromList(windows);
  }

  Float32List _computeMelSpectrogram(Float32List audio) {
    // This is a simplified mel spectrogram computation
    // In practice, you'd use a proper audio processing library like dart_fft

    final spectrogram = <double>[];
    final windowFunction = _generateHannWindow(config.nFFT);

    // Compute STFT frames
    for (int start = 0; start + config.nFFT <= audio.length; start += config.hopLength) {
      // Apply window function
      final windowedFrame = List<double>.generate(config.nFFT, (i) {
        return audio[start + i] * windowFunction[i];
      });

      // Simplified FFT magnitude (in practice, use proper FFT)
      final melFrame = _computeMelFrame(windowedFrame);
      spectrogram.addAll(melFrame);
    }

    return Float32List.fromList(spectrogram);
  }

  List<double> _generateHannWindow(int size) {
    return List.generate(size, (i) {
      return 0.5 * (1 - math.cos(2 * math.pi * i / (size - 1)));
    });
  }

  List<double> _computeMelFrame(List<double> frame) {
    // Simplified mel filter bank application
    // In practice, you'd compute proper mel filter banks and apply FFT

    final melFrame = <double>[];
    final frameSize = frame.length;
    final melBinSize = frameSize ~/ config.nMels;

    for (int mel = 0; mel < config.nMels; mel++) {
      double melValue = 0.0;
      final start = mel * melBinSize;
      final end = math.min(start + melBinSize, frameSize);

      // Simple averaging over frequency bins
      for (int i = start; i < end; i++) {
        melValue += frame[i].abs();
      }

      melValue /= (end - start);
      melFrame.add(math.log(melValue + 1e-8)); // Log mel spectrogram
    }

    return melFrame;
  }
}

/// Postprocessor for audio classification results
class AudioClassificationPostprocessor extends ExecuTorchPostprocessor<AudioClassificationResult> {
  AudioClassificationPostprocessor({
    required this.classLabels,
  });

  final List<String> classLabels;

  @override
  String get outputTypeName => 'Audio Classification Result';

  @override
  bool validateOutputs(List<TensorData> outputs) {
    if (outputs.isEmpty) return false;

    final output = outputs.first;
    if (output.dataType != TensorType.float32) return false;

    // Check if shape represents logits/probabilities
    final shape = output.shape?.where((dim) => dim != null).toList() ?? [];
    if (shape.isEmpty) return false;

    // Should have correct number of classes
    final outputSize = shape.last!;
    return outputSize == classLabels.length;
  }

  @override
  Future<AudioClassificationResult> postprocess(List<TensorData> outputs, {ModelMetadata? metadata}) async {
    try {
      if (outputs.isEmpty) {
        throw PostprocessingException('No output tensors provided');
      }

      final output = outputs.first;
      final logits = ProcessorTensorUtils.extractFloat32Data(output);

      // Apply softmax to get probabilities
      final probabilities = _applySoftmax(logits);

      // Find the class with highest probability
      double maxProb = 0.0;
      int maxIndex = 0;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      // Get class name
      String className;
      if (maxIndex < classLabels.length) {
        className = classLabels[maxIndex];
      } else {
        className = 'Unknown Class $maxIndex';
      }

      // Validate confidence range
      if (maxProb < 0.0 || maxProb > 1.0) {
        throw PostprocessingException(
          'Invalid confidence value: $maxProb (should be between 0.0 and 1.0)'
        );
      }

      return AudioClassificationResult(
        className: className,
        confidence: maxProb,
        classIndex: maxIndex,
        allProbabilities: probabilities,
      );
    } catch (e) {
      if (e is ProcessorException) rethrow;
      throw PostprocessingException('Audio classification postprocessing failed: $e', e);
    }
  }

  List<double> _applySoftmax(Float32List logits) {
    // Find max value for numerical stability
    double maxLogit = logits.reduce(math.max);

    // Compute exp(x - max) for each element
    final expValues = logits.map((x) => math.exp(x - maxLogit)).toList();

    // Compute sum of exponentials
    final sumExp = expValues.reduce((a, b) => a + b);

    // Normalize to get probabilities
    return expValues.map((x) => x / sumExp).toList();
  }
}

/// Complete audio classification processor
class AudioClassificationProcessor extends ExecuTorchProcessor<Float32List, AudioClassificationResult> {
  AudioClassificationProcessor({
    required this.sampleRate,
    required this.windowSize,
    required this.classLabels,
    this.normalizeAudio = true,
    this.applyMelSpectrogram = false,
  }) : config = AudioPreprocessConfig(
         sampleRate: sampleRate,
         windowSize: windowSize,
         normalizeAudio: normalizeAudio,
         applyMelSpectrogram: applyMelSpectrogram,
       ),
       _preprocessor = AudioClassificationPreprocessor(
         config: AudioPreprocessConfig(
           sampleRate: sampleRate,
           windowSize: windowSize,
           normalizeAudio: normalizeAudio,
           applyMelSpectrogram: applyMelSpectrogram,
         ),
       ),
       _postprocessor = AudioClassificationPostprocessor(classLabels: classLabels);

  final int sampleRate;
  final int windowSize;
  final List<String> classLabels;
  final bool normalizeAudio;
  final bool applyMelSpectrogram;
  final AudioPreprocessConfig config;
  final AudioClassificationPreprocessor _preprocessor;
  final AudioClassificationPostprocessor _postprocessor;

  @override
  ExecuTorchPreprocessor<Float32List> get preprocessor => _preprocessor;

  @override
  ExecuTorchPostprocessor<AudioClassificationResult> get postprocessor => _postprocessor;
}

/// Speech command recognition processor (specialized audio classification)
class SpeechCommandProcessor extends AudioClassificationProcessor {
  SpeechCommandProcessor({
    int sampleRate = 16000,
    int windowSize = 1024,
    required List<String> commands,
  }) : super(
    sampleRate: sampleRate,
    windowSize: windowSize,
    classLabels: commands,
    normalizeAudio: true,
    applyMelSpectrogram: true,
  );
}

/// Music genre classification processor (specialized audio classification)
class MusicGenreProcessor extends AudioClassificationProcessor {
  MusicGenreProcessor({
    int sampleRate = 22050,
    int windowSize = 2048,
    required List<String> genres,
  }) : super(
    sampleRate: sampleRate,
    windowSize: windowSize,
    classLabels: genres,
    normalizeAudio: true,
    applyMelSpectrogram: true,
  );
}

/// Environmental sound classification processor
class EnvironmentalSoundProcessor extends AudioClassificationProcessor {
  EnvironmentalSoundProcessor({
    int sampleRate = 16000,
    int windowSize = 1024,
  }) : super(
    sampleRate: sampleRate,
    windowSize: windowSize,
    classLabels: const [
      'silence',
      'speech',
      'music',
      'traffic',
      'nature',
      'machinery',
      'alarm',
      'animal',
      'footsteps',
      'applause',
    ],
    normalizeAudio: true,
    applyMelSpectrogram: true,
  );
}