import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'executorch_flutter_method_channel.dart';

abstract class ExecutorchFlutterPlatform extends PlatformInterface {
  /// Constructs a ExecutorchFlutterPlatform.
  ExecutorchFlutterPlatform() : super(token: _token);

  static final Object _token = Object();

  static ExecutorchFlutterPlatform _instance = MethodChannelExecutorchFlutter();

  /// The default instance of [ExecutorchFlutterPlatform] to use.
  ///
  /// Defaults to [MethodChannelExecutorchFlutter].
  static ExecutorchFlutterPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [ExecutorchFlutterPlatform] when
  /// they register themselves.
  static set instance(ExecutorchFlutterPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
