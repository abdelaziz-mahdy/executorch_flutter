import 'dart:io' show Platform;

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:universal_platform/universal_platform.dart';

import '../models/model_definition.dart';
import '../models/model_registry.dart';
import '../services/model_controller.dart';
import '../services/model_download_service.dart';
import '../ui/widgets/performance_monitor.dart';

/// Unified Model Playground - works with any model type through ModelDefinition
class UnifiedModelPlayground extends StatefulWidget {
  const UnifiedModelPlayground({super.key});

  @override
  State<UnifiedModelPlayground> createState() => _UnifiedModelPlaygroundState();
}

class _UnifiedModelPlaygroundState extends State<UnifiedModelPlayground> {
  // Model state
  List<ModelDefinition>? _availableModels;
  ModelController? _controller;

  // Version state
  List<String> _availableVersions = [executorchVersion];
  String _selectedVersion = executorchVersion;

  // Loading state
  bool _isLoadingModels = true;
  bool _isLoadingModel = false;

  // Download state
  double _downloadProgress = 0.0;
  bool _isDownloading = false;

  // UI state
  bool _isInputExpanded = true;

  @override
  void initState() {
    super.initState();
    _loadVersionsAndModels();
  }

  Future<void> _loadVersionsAndModels() async {
    // First load available versions
    try {
      final versions = await ModelRegistry.fetchAvailableVersions();
      setState(() {
        _availableVersions = versions.versions;
        // Keep selected version if it's still valid, otherwise use latest
        if (!_availableVersions.contains(_selectedVersion)) {
          _selectedVersion = versions.latest;
        }
      });
    } catch (e) {
      debugPrint('Failed to load versions: $e');
      // Keep default version
    }

    // Then load models for selected version
    await _loadAvailableModels();
  }

  Future<void> _onVersionChanged(String version) async {
    if (version == _selectedVersion) return;

    // Dispose current model if any
    final oldController = _controller;
    if (oldController != null) {
      oldController.removeListener(_onControllerChanged);
      setState(() {
        _controller = null;
      });
      await oldController.dispose();
    }

    setState(() {
      _selectedVersion = version;
      _isLoadingModels = true;
      _availableModels = null;
    });

    // Update the service's selected version
    ModelRegistry.selectedVersion = version;

    // Reload models for the new version
    await _loadAvailableModels();
  }

  void _onControllerChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    _controller?.removeListener(_onControllerChanged);
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _loadAvailableModels() async {
    try {
      final models = await ModelRegistry.loadAll(version: _selectedVersion);
      setState(() {
        _availableModels = models;
        _isLoadingModels = false;
      });
    } catch (e) {
      debugPrint('Failed to load models: $e');
      setState(() {
        _isLoadingModels = false;
      });
    }
  }

  Future<void> _selectModel(ModelDefinition model) async {
    // Remove listener and dispose previous controller (handles camera cleanup)
    final oldController = _controller;
    if (oldController != null) {
      oldController.removeListener(_onControllerChanged);
    }

    setState(() {
      _controller = null; // Clear controller immediately to avoid stale state
      _isLoadingModel = true;
      _isDownloading = false;
      _downloadProgress = 0.0;
    });

    // Dispose after clearing reference to prevent race conditions
    await oldController?.dispose();

    try {
      // Download model from remote URL (or use cached version)
      final downloadService = ModelDownloadService.instance;

      // Check if model needs to be downloaded (version-aware)
      final isCached = await downloadService.isModelCached(
        model.name,
        version: _selectedVersion,
      );
      if (!isCached) {
        setState(() {
          _isDownloading = true;
          _downloadProgress = 0.0;
        });
      }

      // Download with version-aware caching and hash verification
      final downloadInfo = await downloadService.downloadModel(
        modelName: model.name,
        remoteUrl: model.remoteUrl,
        version: _selectedVersion,
        expectedHash: model.hash,
        onProgress: (progress, received, total) {
          if (mounted) {
            setState(() {
              _downloadProgress = progress;
            });
          }
        },
      );

      if (downloadInfo.state == ModelDownloadState.error) {
        throw Exception(downloadInfo.errorMessage ?? 'Download failed');
      }

      // Log model hash for debugging cache issues
      if (downloadInfo.bytes != null) {
        final actualHash = CachedModelDataSource.computeHash(
          downloadInfo.bytes!,
        );
        debugPrint(
          '🔑 Model hash: $actualHash\n'
          '   Expected:   ${model.hash}\n'
          '   Match: ${actualHash == model.hash}',
        );
      }

      setState(() {
        _isDownloading = false;
      });

      // Load the model
      final ExecuTorchModel execuTorchModel;
      if (UniversalPlatform.isWeb) {
        // Web: Load from downloaded bytes (in memory)
        execuTorchModel = await ExecuTorchModel.loadFromBytes(
          downloadInfo.bytes!,
        );
      } else {
        // Native: Load from cached file path
        execuTorchModel = await ExecuTorchModel.load(downloadInfo.localPath!);
      }
      final settings = model.createDefaultSettings();

      final controller = await ModelController.create(
        definition: model,
        execuTorchModel: execuTorchModel,
        settings: settings,
      );

      if (mounted) {
        setState(() {
          _controller = controller;
          _controller!.addListener(_onControllerChanged);
          _isLoadingModel = false;
        });
      } else {
        // Widget was unmounted during loading, clean up
        await controller.dispose();
      }
    } catch (e) {
      debugPrint('❌ Failed to load model: $e');
      if (mounted) {
        setState(() {
          _controller = null; // Ensure controller is null on failure
          _isLoadingModel = false;
          _isDownloading = false;
        });

        // Show download error or model load error
        final errorString = e.toString();
        if (errorString.contains('Failed to download') ||
            errorString.contains('HTTP')) {
          _showDownloadError(model, errorString);
        } else {
          _showModelLoadError(model, errorString);
        }
      }
    }
  }

  void _showDownloadError(ModelDefinition model, String error) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(Icons.cloud_off, color: Theme.of(context).colorScheme.error),
            const SizedBox(width: 8),
            const Text('Download Failed'),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Failed to download ${model.displayName}',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              Text('Model URL:', style: Theme.of(context).textTheme.labelSmall),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4, bottom: 16),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  model.remoteUrl,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(fontFamily: 'monospace'),
                ),
              ),
              Text('Error:', style: Theme.of(context).textTheme.labelSmall),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  error,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                    color: Theme.of(context).colorScheme.onErrorContainer,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.info_outline,
                      size: 16,
                      color: Theme.of(context).colorScheme.onPrimaryContainer,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Check your internet connection and try again. Models are downloaded from GitHub on first use.',
                        style: TextStyle(
                          fontSize: 12,
                          color: Theme.of(
                            context,
                          ).colorScheme.onPrimaryContainer,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showModelLoadError(ModelDefinition model, String error) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            Icon(
              Icons.error_outline,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(width: 8),
            const Text('Failed to Load Model'),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Failed to load ${model.displayName}',
                style: Theme.of(
                  context,
                ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              Text(
                'Error details:',
                style: Theme.of(context).textTheme.labelSmall,
              ),
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  error,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                    color: Theme.of(context).colorScheme.error,
                  ),
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<Map<String, String>> _getDeviceInfo() async {
    final info = <String, String>{};
    if (UniversalPlatform.isWeb) {
      info['Platform'] = 'Web';
      return info;
    }
    final deviceInfo = DeviceInfoPlugin();
    if (Platform.isAndroid) {
      final android = await deviceInfo.androidInfo;
      info['Platform'] = 'Android';
      info['Device'] = android.model;
      info['Manufacturer'] = android.manufacturer;
      info['Brand'] = android.brand;
      info['Hardware'] = android.hardware;
      info['Board'] = android.board;
      info['Product'] = android.product;
      info['Android Version'] = android.version.release;
      info['SDK Int'] = '${android.version.sdkInt}';
      info['Security Patch'] = android.version.securityPatch ?? 'N/A';
      info['ABIs'] = android.supportedAbis.join(', ');
      info['Physical RAM'] = '${android.physicalRamSize} MB';
      info['Available RAM'] = '${android.availableRamSize} MB';
      info['Low RAM Device'] = '${android.isLowRamDevice}';
      info['Physical Device'] = '${android.isPhysicalDevice}';
    } else if (Platform.isIOS) {
      final ios = await deviceInfo.iosInfo;
      info['Platform'] = 'iOS';
      info['Device'] = ios.model;
      info['Name'] = ios.name;
      info['System Version'] = ios.systemVersion;
      info['Machine'] = ios.utsname.machine;
      info['Physical Device'] = '${ios.isPhysicalDevice}';
    } else if (Platform.isMacOS) {
      final macos = await deviceInfo.macOsInfo;
      info['Platform'] = 'macOS';
      info['Model'] = macos.model;
      info['OS Version'] = macos.osRelease;
      info['Kernel Version'] = macos.kernelVersion;
      info['CPU Architecture'] = macos.arch;
      info['Physical RAM'] = '${macos.memorySize ~/ (1024 * 1024)} MB';
      info['CPU Cores'] = '${macos.activeCPUs}';
    } else if (Platform.isLinux) {
      final linux = await deviceInfo.linuxInfo;
      info['Platform'] = 'Linux';
      info['Name'] = linux.prettyName;
      info['Version'] = linux.version ?? 'N/A';
      info['Machine'] = linux.machineId ?? 'N/A';
    } else if (Platform.isWindows) {
      final windows = await deviceInfo.windowsInfo;
      info['Platform'] = 'Windows';
      info['Computer Name'] = windows.computerName;
      info['Product Name'] = windows.productName;
      info['Build Number'] = '${windows.buildNumber}';
      info['Physical RAM'] = '${windows.systemMemoryInMegabytes} MB';
      info['CPU Cores'] = '${windows.numberOfCores}';
    }
    return info;
  }

  String _buildRuntimeInfoText(Map<String, String> deviceInfo) {
    final buf = StringBuffer();
    final availableBackends = BackendQuery.available;
    final allBackends = Backend.values;
    final ffiVersion = ExecuTorchVersion.version;
    final etVersion = ExecuTorchVersion.executorchVersion;

    buf.writeln('## ExecuTorch Runtime Info\n');

    buf.writeln('### Version');
    buf.writeln('- ExecuTorch: $etVersion');
    buf.writeln('- FFI Library: $ffiVersion');
    buf.writeln('- Plugin: $executorchVersion');
    buf.writeln('- Selected Version: $_selectedVersion\n');

    buf.writeln('### Backends');
    for (final backend in allBackends) {
      final available = availableBackends.contains(backend);
      buf.writeln(
        '- ${backend.displayName}: ${available ? "Available" : "Not compiled"}',
      );
    }
    buf.writeln();

    buf.writeln('### Device');
    for (final entry in deviceInfo.entries) {
      buf.writeln('- ${entry.key}: ${entry.value}');
    }

    return buf.toString();
  }

  void _showRuntimeInfoDialog() async {
    // Show loading dialog while fetching device info
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const AlertDialog(
        content: Row(
          children: [
            CircularProgressIndicator(),
            SizedBox(width: 16),
            Text('Loading device info...'),
          ],
        ),
      ),
    );

    final deviceInfo = await _getDeviceInfo();
    final runtimeInfoText = _buildRuntimeInfoText(deviceInfo);
    debugPrint(runtimeInfoText);

    if (!mounted) return;
    Navigator.of(context).pop(); // dismiss loading

    // Get available backends
    final availableBackends = BackendQuery.available;
    final allBackends = Backend.values;

    // Get version info
    final ffiVersion = ExecuTorchVersion.version;
    final etVersion = ExecuTorchVersion.executorchVersion;

    if (!mounted) return;
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.info_outline),
            SizedBox(width: 8),
            Text('Runtime Info'),
          ],
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Version section
              Text(
                'Version',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              _buildInfoRow('ExecuTorch', etVersion),
              _buildInfoRow('FFI Library', ffiVersion),
              _buildInfoRow('Plugin', executorchVersion),
              const Divider(height: 24),

              // Backends section
              Text(
                'Backends',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ...allBackends.map((backend) {
                final isAvailable = availableBackends.contains(backend);
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(
                        isAvailable ? Icons.check_circle : Icons.cancel,
                        color: isAvailable ? Colors.green : Colors.red,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Text(backend.displayName),
                      const Spacer(),
                      Text(
                        isAvailable ? 'Available' : 'Not compiled',
                        style: TextStyle(
                          color: isAvailable ? Colors.green : Colors.grey,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                );
              }),
              const Divider(height: 24),

              // Device section
              Text(
                'Device',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ...deviceInfo.entries.map((e) => _buildInfoRow(e.key, e.value)),
              const Divider(height: 24),

              _buildInfoRow('Selected Version', _selectedVersion),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              final text = _buildRuntimeInfoText(deviceInfo);
              Clipboard.setData(ClipboardData(text: text));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Runtime info copied to clipboard'),
                ),
              );
            },
            child: const Text('Copy'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  void _showSettingsDialog() {
    if (_controller == null) return;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        expand: false,
        builder: (context, scrollController) => ListenableBuilder(
          listenable: _controller!,
          builder: (context, _) => SingleChildScrollView(
            controller: scrollController,
            child: Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom + 16,
                top: 16,
                left: 8,
                right: 8,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Drag handle
                  Container(
                    width: 40,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  // Settings content
                  _controller!.buildSettingsWidget(context),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Playground'),
        elevation: 0,
        actions: [
          // Info button - shows runtime info (backends, version)
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showRuntimeInfoDialog,
            tooltip: 'Runtime Info',
          ),
          // Settings button - only shown when a model is selected
          if (_controller != null)
            IconButton(
              icon: const Icon(Icons.settings),
              onPressed: _showSettingsDialog,
              tooltip: 'Model Settings',
            ),
        ],
      ),
      body: _isLoadingModels
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                // Model selector
                _buildModelSelector(),

                // Main content
                Expanded(
                  child: _controller == null
                      ? _buildEmptyState()
                      : LayoutBuilder(
                          builder: (context, constraints) {
                            final isLargeScreen = constraints.maxWidth > 900;

                            if (isLargeScreen) {
                              // Horizontal layout for large screens
                              return Row(
                                children: [
                                  // Left: Result display (60% width)
                                  Expanded(
                                    flex: 6,
                                    child: _buildResultSection(
                                      isLargeScreen: true,
                                    ),
                                  ),

                                  // Right: Input + Details (40% width)
                                  Expanded(flex: 4, child: _buildSidePanel()),
                                ],
                              );
                            } else {
                              // Vertical layout for mobile/small screens
                              return Stack(
                                children: [
                                  // Result display area (full screen, scrollable)
                                  _buildResultSection(isLargeScreen: false),

                                  // Collapsible input panel at bottom
                                  Positioned(
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    child: _buildInputSection(),
                                  ),

                                  // Toggle button
                                  if (!_isInputExpanded && _controller != null)
                                    Positioned(
                                      right: 16,
                                      bottom: 16,
                                      child: FloatingActionButton(
                                        onPressed: () {
                                          setState(() {
                                            _isInputExpanded = true;
                                          });
                                        },
                                        child: const Icon(Icons.input),
                                      ),
                                    ),
                                ],
                              );
                            }
                          },
                        ),
                ),
              ],
            ),
    );
  }

  Widget _buildModelSelector() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Version selector row
          Row(
            children: [
              // Version dropdown (compact)
              SizedBox(
                width: 120,
                child: DropdownButtonFormField<String>(
                  initialValue: _selectedVersion,
                  decoration: InputDecoration(
                    labelText: 'Version',
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  items: _availableVersions.map((version) {
                    final isCurrentPlugin = version == executorchVersion;
                    return DropdownMenuItem(
                      value: version,
                      child: Text(
                        isCurrentPlugin ? '$version ✓' : version,
                        style: TextStyle(
                          fontWeight: isCurrentPlugin
                              ? FontWeight.bold
                              : FontWeight.normal,
                        ),
                      ),
                    );
                  }).toList(),
                  onChanged: _isLoadingModel || _isLoadingModels
                      ? null
                      : (version) {
                          if (version != null) _onVersionChanged(version);
                        },
                ),
              ),
              const SizedBox(width: 12),
              // Model dropdown (expanded)
              Expanded(
                child: DropdownButtonFormField<ModelDefinition>(
                  initialValue: _controller?.definition,
                  decoration: InputDecoration(
                    labelText: 'Select Model',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  items: _availableModels?.map((model) {
                    return DropdownMenuItem(
                      value: model,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(model.icon, size: 20),
                          const SizedBox(width: 12),
                          Flexible(
                            fit: FlexFit.loose,
                            child: Text(
                              model.fileSizeMB > 0
                                  ? '${model.displayName} (${model.fileSizeMB.toStringAsFixed(1)} MB)'
                                  : model.displayName,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                  onChanged: _isLoadingModel
                      ? null
                      : (model) {
                          if (model != null) _selectModel(model);
                        },
                ),
              ),
            ],
          ),
          // Version info hint
          if (_selectedVersion != executorchVersion)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 16,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Using models for ExecuTorch $_selectedVersion (plugin built with $executorchVersion)',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    // Show download/loading progress
    if (_isLoadingModel) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_isDownloading) ...[
              Icon(
                Icons.cloud_download,
                size: 64,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 16),
              Text(
                'Downloading model...',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Text(
                '${(_downloadProgress * 100).toStringAsFixed(0)}%',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: 200,
                child: LinearProgressIndicator(
                  value: _downloadProgress,
                  backgroundColor: Theme.of(
                    context,
                  ).colorScheme.surfaceContainerHighest,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Models are downloaded from GitHub on first use',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ] else ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Loading model...',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ],
          ],
        ),
      );
    }

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.model_training,
            size: 64,
            color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.5),
          ),
          const SizedBox(height: 16),
          Text(
            'Select a model to get started',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          Text(
            'Models are downloaded from GitHub on first use',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInputSection() {
    if (_controller == null || !_isInputExpanded) return const SizedBox();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.input,
                size: 18,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              const SizedBox(width: 8),
              Text(
                'Input',
                style: Theme.of(
                  context,
                ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.close, size: 20),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: () {
                  setState(() {
                    _isInputExpanded = false;
                  });
                },
                tooltip: 'Hide',
              ),
            ],
          ),
          const SizedBox(height: 8),
          _controller!.buildInputWidget(
            context: context,
            onInputSelected: (input) => _controller?.processInput(input),
          ),
        ],
      ),
    );
  }

  Widget _buildResultSection({bool isLargeScreen = false}) {
    if (_controller?.isCameraMode ?? false) {
      return _buildCameraSection();
    }

    if (_controller?.isProcessing ?? false) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Processing...'),
          ],
        ),
      );
    }

    final errorMessage = _controller?.errorMessage;
    if (errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Card(
            color: Theme.of(context).colorScheme.errorContainer,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Icon(
                    Icons.error_outline,
                    color: Theme.of(context).colorScheme.onErrorContainer,
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      errorMessage,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onErrorContainer,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    final input = _controller?.currentInput;
    if (input == null) {
      return Center(
        child: Text(
          'Select an input to see results',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }

    final result = _controller?.currentResult;

    if (isLargeScreen) {
      return Container(
        padding: const EdgeInsets.all(16),
        child: Center(
          child: _controller!.buildResultRenderer(
            context: context,
            input: input,
            result: result,
          ),
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          SizedBox(
            height: 400,
            child: _controller!.buildResultRenderer(
              context: context,
              input: input,
              result: result,
            ),
          ),

          if (result != null) _buildDetailsSection(),

          if (_isInputExpanded) const SizedBox(height: 150),
        ],
      ),
    );
  }

  Widget _buildCameraSection() {
    final input = _controller?.currentInput;
    final result = _controller?.currentResult;
    final showPerformance =
        _controller?.settings.showPerformanceOverlay ?? true;
    final performanceMetrics = _controller?.performanceMetrics;

    Widget cameraContent;
    if (input == null) {
      cameraContent = const Center(child: CircularProgressIndicator());
    } else {
      cameraContent = RepaintBoundary(
        child: _controller!.buildResultRenderer(
          context: context,
          input: input,
          result: result,
        ),
      );
    }

    return Stack(
      children: [
        cameraContent,
        if (showPerformance && (performanceMetrics?.hasData ?? false))
          Positioned(
            top: 16,
            right: 16,
            child: _controller!.definition.buildPerformanceMonitor(
              context: context,
              metrics: performanceMetrics!,
              displayMode: PerformanceDisplayMode.overlay,
            ),
          ),
      ],
    );
  }

  Widget _buildDetailsSection() {
    final showPerformance =
        !(_controller?.isCameraMode ?? false) &&
        (_controller?.settings.showPerformanceOverlay ?? true);
    final performanceMetrics = _controller?.performanceMetrics;
    final result = _controller?.currentResult;

    if (result == null) return const SizedBox();

    return Column(
      children: [
        if (showPerformance && (performanceMetrics?.hasData ?? false))
          _controller!.definition.buildPerformanceMonitor(
            context: context,
            metrics: performanceMetrics!,
            displayMode: PerformanceDisplayMode.section,
          ),

        if (showPerformance && (performanceMetrics?.hasData ?? false))
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: const Divider(),
          ),

        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.check_circle,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(width: 12),
                  Text(
                    'Results',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _controller!.definition.buildResultsDetailsSection(
                context: context,
                result: result,
                processingTime: performanceMetrics?.totalTime,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSidePanel() {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        border: Border(
          left: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
        ),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Input section (always visible on large screens)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surface,
                border: Border(
                  bottom: BorderSide(
                    color: Theme.of(context).colorScheme.outlineVariant,
                  ),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Input',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _controller!.buildInputWidget(
                    context: context,
                    onInputSelected: (input) =>
                        _controller?.processInput(input),
                  ),
                ],
              ),
            ),

            // Results details section
            if (_controller?.currentResult != null) _buildDetailsSection(),
          ],
        ),
      ),
    );
  }
}
