import 'model_settings.dart';

/// Settings specific to Gemma text generation model
class GemmaModelSettings extends ModelSettings {
  GemmaModelSettings({
    super.showPerformanceOverlay,
    int maxLength = 128,
    double temperature = 0.7,
    double topP = 0.9,
    int topK = 50,
  }) : _maxLength = maxLength,
       _temperature = temperature,
       _topP = topP,
       _topK = topK;

  /// Maximum number of tokens to generate (32-512)
  int _maxLength;
  int get maxLength => _maxLength;
  set maxLength(int value) {
    if (_maxLength != value) {
      _maxLength = value.clamp(32, 512);
      notifyListeners();
    }
  }

  /// Temperature for sampling (0.1-2.0)
  /// Lower values = more deterministic, higher values = more creative
  double _temperature;
  double get temperature => _temperature;
  set temperature(double value) {
    if (_temperature != value) {
      _temperature = value.clamp(0.1, 2.0);
      notifyListeners();
    }
  }

  /// Top-p (nucleus sampling) threshold (0.1-1.0)
  /// Only tokens with cumulative probability <= top_p are considered
  double _topP;
  double get topP => _topP;
  set topP(double value) {
    if (_topP != value) {
      _topP = value.clamp(0.1, 1.0);
      notifyListeners();
    }
  }

  /// Top-k sampling - only consider top K tokens (1-100)
  int _topK;
  int get topK => _topK;
  set topK(int value) {
    if (_topK != value) {
      _topK = value.clamp(1, 100);
      notifyListeners();
    }
  }

  @override
  GemmaModelSettings copyWith({
    bool? showPerformanceOverlay,
    int? maxLength,
    double? temperature,
    double? topP,
    int? topK,
  }) {
    return GemmaModelSettings(
      showPerformanceOverlay:
          showPerformanceOverlay ?? this.showPerformanceOverlay,
      maxLength: maxLength ?? _maxLength,
      temperature: temperature ?? _temperature,
      topP: topP ?? _topP,
      topK: topK ?? _topK,
    );
  }

  @override
  void reset() {
    showPerformanceOverlay = true;
    maxLength = 128;
    temperature = 0.7;
    topP = 0.9;
    topK = 50;
  }
}
