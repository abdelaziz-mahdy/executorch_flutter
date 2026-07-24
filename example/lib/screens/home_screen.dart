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
      appBar: AppBar(title: const Text('ExecuTorch Examples'), elevation: 0),
      // Cards size to their content and stay centred in a readable column
      // rather than stretching to fill a desktop-sized window. The min-height
      // constraint keeps them vertically centred while still allowing scroll
      // on short viewports.
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, viewport) {
            return SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: viewport.maxHeight - 48),
                child: Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 840),
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        const gap = 20.0;
                        // Two across when there's room, stacked when narrow.
                        final columns = constraints.maxWidth >= 620 ? 2 : 1;
                        final cardWidth =
                            (constraints.maxWidth - gap * (columns - 1)) /
                            columns;
                        return Wrap(
                          spacing: gap,
                          runSpacing: gap,
                          alignment: WrapAlignment.center,
                          children: [
                            for (final category in _categories)
                              SizedBox(
                                width: cardWidth,
                                child: _CategoryCard(category: category),
                              ),
                          ],
                        );
                      },
                    ),
                  ),
                ),
              ),
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
        onTap: () => Navigator.of(
          context,
        ).push(MaterialPageRoute<void>(builder: category.builder)),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
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
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 6),
              Text(
                category.subtitle,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 20),
              Align(
                alignment: Alignment.bottomRight,
                child: Icon(
                  Icons.arrow_forward,
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
