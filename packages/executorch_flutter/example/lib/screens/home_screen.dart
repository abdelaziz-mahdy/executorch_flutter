// ExecuTorch Flutter Example - category landing screen.
//
// The example app spans fundamentally different model interaction styles:
// single-shot vision inference (driven by the ModelDefinition strategy +
// UnifiedModelPlayground) and stateful streaming LLM chat (driven by
// ExecuTorchLLM). Rather than force both through one abstraction, this home
// screen splits the app into categories, each routing to the screen that fits
// its interaction model. Add a new [_Category] to surface more demos.
library;

import 'package:material_ui/material_ui.dart';

import '../llm/llm_files_stub.dart'
    if (dart.library.io) '../llm/llm_files_io.dart'
    if (dart.library.js_interop) '../llm/llm_files_web.dart'
    show llmSupported;
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
    this.unavailable,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final WidgetBuilder builder;

  /// Why this category can't run on this platform; null when it can.
  final String? unavailable;
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
      subtitle: 'On-device LLM streaming chat\n(Qwen3, or bring your own)',
      icon: Icons.chat_bubble_outline,
      color: Color(0xFF34A853),
      builder: _buildChat,
      unavailable: llmSupported ? null : 'Not available on this platform',
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
    final unavailable = category.unavailable;
    final card = Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: unavailable != null
            ? null
            : () => Navigator.of(
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
                child: unavailable == null
                    ? Icon(Icons.arrow_forward, color: scheme.onSurfaceVariant)
                    : Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.block,
                            size: 16,
                            color: scheme.onSurfaceVariant,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            unavailable,
                            style: Theme.of(context).textTheme.labelLarge
                                ?.copyWith(color: scheme.onSurfaceVariant),
                          ),
                        ],
                      ),
              ),
            ],
          ),
        ),
      ),
    );
    // Greyed out and not tappable where the demo can't run.
    return unavailable == null ? card : Opacity(opacity: 0.55, child: card);
  }
}
