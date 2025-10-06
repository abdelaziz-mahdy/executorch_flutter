import 'package:flutter/material.dart';

/// Base interface for rendering model results
/// Generic and reusable across different input types (image, text, audio, etc.)
abstract class BaseResultRenderer<TInput, TResult> extends StatelessWidget {
  const BaseResultRenderer({
    super.key,
    required this.input,
    required this.result,
  });

  final TInput input;
  final TResult? result;

  @override
  Widget build(BuildContext context);
}
