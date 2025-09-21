
import 'executorch_flutter_platform_interface.dart';

class ExecutorchFlutter {
  Future<String?> getPlatformVersion() {
    return ExecutorchFlutterPlatform.instance.getPlatformVersion();
  }
}
