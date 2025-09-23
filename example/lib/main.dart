/// ExecuTorch Flutter Example App
///
/// A comprehensive demonstration app showcasing ExecuTorch integration with Flutter,
/// featuring the new processor interfaces for different machine learning tasks.
///
/// Features:
/// - Image Classification with ImageNetProcessor
/// - Text Classification with SentimentAnalysisProcessor
/// - Audio Classification with EnvironmentalSoundProcessor
/// - Performance metrics and error handling
/// - Modern Material 3 design with tabbed interface
library;

import 'package:flutter/material.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

// Import enhanced demo pages
import 'screens/enhanced_image_demo.dart';
import 'screens/enhanced_text_demo.dart';
import 'screens/enhanced_audio_demo.dart';
import 'screens/camera_demo.dart';

// Import services and widgets
import 'services/performance_service.dart';
import 'widgets/performance_monitor.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize ExecuTorch manager
  try {
    await ExecutorchManager.instance.initialize();
    debugPrint('✅ ExecuTorch Manager initialized successfully');
  } catch (e) {
    debugPrint('❌ Failed to initialize ExecuTorch: $e');
  }

  // Initialize Performance Service
  try {
    await PerformanceService().initialize();
    debugPrint('✅ Performance Service initialized successfully');
  } catch (e) {
    debugPrint('❌ Failed to initialize Performance Service: $e');
  }

  runApp(const ExecuTorchProcessorExampleApp());
}

class ExecuTorchProcessorExampleApp extends StatelessWidget {
  const ExecuTorchProcessorExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ExecuTorch Processor Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const ProcessorDemoHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ProcessorDemoHomePage extends StatefulWidget {
  const ProcessorDemoHomePage({super.key});

  @override
  State<ProcessorDemoHomePage> createState() => _ProcessorDemoHomePageState();
}

class _ProcessorDemoHomePageState extends State<ProcessorDemoHomePage>
    with TickerProviderStateMixin {
  late TabController _tabController;

  final List<ProcessorTab> _tabs = [
    ProcessorTab(
      icon: Icons.image,
      label: 'Image',
      description: 'Image Classification with camera, gallery, and real-time processing',
      widget: const EnhancedImageDemo(),
    ),
    ProcessorTab(
      icon: Icons.text_fields,
      label: 'Text',
      description: 'Text analysis: sentiment, topic classification, and language detection',
      widget: const EnhancedTextDemo(),
    ),
    ProcessorTab(
      icon: Icons.audiotrack,
      label: 'Audio',
      description: 'Audio analysis: sound classification, emotion, and music genre detection',
      widget: const EnhancedAudioDemo(),
    ),
    ProcessorTab(
      icon: Icons.camera_alt,
      label: 'Camera',
      description: 'Real-time Camera Processing and ML Inference with object detection',
      widget: const CameraDemo(),
    ),
  ];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: _tabs.length, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('ExecuTorch Processor Demo'),
        backgroundColor: theme.colorScheme.inversePrimary,
        elevation: 0,
        bottom: TabBar(
          controller: _tabController,
          tabs: _tabs.map((tab) => Tab(
            icon: Icon(tab.icon),
            text: tab.label,
          )).toList(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showInfoDialog(context),
            tooltip: 'About this demo',
          ),
        ],
      ),
      body: Column(
        children: [
          // Header with current tab description
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16.0),
            color: theme.colorScheme.surfaceVariant.withOpacity(0.5),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: Text(
                _tabs[_tabController.index].description,
                key: ValueKey(_tabController.index),
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),

          // Tab content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: _tabs.map((tab) => tab.widget).toList(),
            ),
          ),

          // Performance monitor (persistent across tabs)
          const PerformanceMonitor(),
        ],
      ),
    );
  }

  void _showInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('ExecuTorch Processor Demo'),
        content: const SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'This app demonstrates the new ExecuTorch processor interfaces:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              _InfoItem(
                icon: Icons.image,
                title: 'Image Classification',
                description: 'Uses ImageNetProcessor for preprocessing images and classifying objects using pre-trained models.',
              ),
              SizedBox(height: 12),
              _InfoItem(
                icon: Icons.text_fields,
                title: 'Text Classification',
                description: 'Uses SentimentAnalysisProcessor for tokenizing text and determining sentiment.',
              ),
              SizedBox(height: 12),
              _InfoItem(
                icon: Icons.audiotrack,
                title: 'Audio Classification',
                description: 'Uses EnvironmentalSoundProcessor for analyzing audio and classifying sounds.',
              ),
              SizedBox(height: 12),
              _InfoItem(
                icon: Icons.camera_alt,
                title: 'Real-time Camera',
                description: 'Real-time camera processing with ML inference and performance monitoring.',
              ),
              SizedBox(height: 16),
              Text(
                'Each processor handles the complete pipeline from input preprocessing to result postprocessing.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
              SizedBox(height: 8),
              Text(
                'Performance monitoring tracks FPS, processing times, and device capabilities.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Got it'),
          ),
        ],
      ),
    );
  }
}

class _InfoItem extends StatelessWidget {
  const _InfoItem({
    required this.icon,
    required this.title,
    required this.description,
  });

  final IconData icon;
  final String title;
  final String description;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                description,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class ProcessorTab {
  const ProcessorTab({
    required this.icon,
    required this.label,
    required this.description,
    required this.widget,
  });

  final IconData icon;
  final String label;
  final String description;
  final Widget widget;
}