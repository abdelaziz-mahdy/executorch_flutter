import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'executorch_flutter_platform_interface.dart';

/// An implementation of [ExecutorchFlutterPlatform] that uses method channels.
class MethodChannelExecutorchFlutter extends ExecutorchFlutterPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('executorch_flutter');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
