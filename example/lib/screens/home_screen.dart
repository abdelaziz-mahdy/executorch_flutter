// ExecuTorch Flutter Example - category landing screen.
//
// The example app spans fundamentally different model interaction styles:
// single-shot vision inference (driven by the ModelDefinition strategy +
// UnifiedModelPlayground) and stateful streaming LLM chat (driven by
// ExecuTorchLLM). Rather than force both through one abstraction, this home
// screen splits the app into categories, each routing to the screen that fits
// its interaction model. Add a new [_Category] to surface more demos.
library;

import 'package:flutter/material.dart';

import 'llm_chat_screen.dart';
import 'unified_model_playground.dart';

/// A top-level example category shown on the home screen.
class _Category {
  const _Category({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.builder,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final WidgetBuilder builder;
}

/// Landing screen that splits the example app by model category.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  static const _categories = <_Category>[
    _Category(
      title: 'Vision Models',
      subtitle: 'Image classification & object detection\n(MobileNet, YOLO)',
      icon: Icons.image_search,
      color: Color(0xFF4285F4),
      builder: _buildPlayground,
    ),
    _Category(
      title: 'Language Models',
      subtitle: 'On-device LLM streaming chat\n(Gemma 4)',
      icon: Icons.chat_bubble_outline,
      color: Color(0xFF34A853),
      builder: _buildChat,
    ),
  ];

  static Widget _buildPlayground(BuildContext context) =>
      const UnifiedModelPlayground();
  static Widget _buildChat(BuildContext context) => const LlmChatScreen();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ExecuTorch Examples'),
        elevation: 0,
      ),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            // One column on narrow screens, two when there's room.
            final crossAxisCount = constraints.maxWidth >= 720 ? 2 : 1;
            return GridView.count(
              crossAxisCount: crossAxisCount,
              padding: const EdgeInsets.all(16),
              mainAxisSpacing: 16,
              crossAxisSpacing: 16,
              childAspectRatio: crossAxisCount == 1 ? 2.4 : 1.6,
              children: [
                for (final category in _categories)
                  _CategoryCard(category: category),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _CategoryCard extends StatelessWidget {
  const _CategoryCard({required this.category});

  final _Category category;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => Navigator.of(context).push(
          MaterialPageRoute<void>(builder: category.builder),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: category.color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(category.icon, color: category.color, size: 28),
              ),
              const SizedBox(height: 16),
              Text(
                category.title,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const SizedBox(height: 6),
              Text(
                category.subtitle,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: scheme.onSurfaceVariant,
                    ),
              ),
              const Spacer(),
              Align(
                alignment: Alignment.bottomRight,
                child: Icon(Icons.arrow_forward, color: scheme.onSurfaceVariant),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
