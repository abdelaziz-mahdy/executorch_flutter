import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_flutter/executorch_flutter_platform_interface.dart';
import 'package:executorch_flutter/executorch_flutter_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockExecutorchFlutterPlatform
    with MockPlatformInterfaceMixin
    implements ExecutorchFlutterPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final ExecutorchFlutterPlatform initialPlatform = ExecutorchFlutterPlatform.instance;

  test('$MethodChannelExecutorchFlutter is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelExecutorchFlutter>());
  });

  test('getPlatformVersion', () async {
    ExecutorchFlutter executorchFlutterPlugin = ExecutorchFlutter();
    MockExecutorchFlutterPlatform fakePlatform = MockExecutorchFlutterPlatform();
    ExecutorchFlutterPlatform.instance = fakePlatform;

    expect(await executorchFlutterPlugin.getPlatformVersion(), '42');
  });
}
