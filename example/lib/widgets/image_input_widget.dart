import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../utils/test_images.dart';

/// Generic reusable image input widget for any image-based model
class ImageInputWidget extends StatelessWidget {
  const ImageInputWidget({
    super.key,
    required this.onImageSelected,
    this.onCameraModeToggle,
    this.isCameraMode = false,
  });

  final Function(File) onImageSelected;
  final VoidCallback? onCameraModeToggle;
  final bool isCameraMode;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _buildActionButton(
            context: context,
            icon: Icons.photo_library,
            label: 'Gallery',
            onTap: () => _pickImage(ImageSource.gallery),
            isEnabled: !isCameraMode,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _buildActionButton(
            context: context,
            icon: Icons.image,
            label: 'Test Image',
            onTap: () => _showTestImagePicker(context),
            isEnabled: !isCameraMode,
          ),
        ),
        if (onCameraModeToggle != null) ...[
          const SizedBox(width: 8),
          Expanded(
            child: _buildActionButton(
              context: context,
              icon: isCameraMode ? Icons.photo : Icons.videocam,
              label: isCameraMode ? 'Image' : 'Camera',
              onTap: onCameraModeToggle!,
              isActive: isCameraMode,
            ),
          ),
        ],
      ],
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      onImageSelected(File(pickedFile.path));
    }
  }

  void _showTestImagePicker(BuildContext context) {
    showModalBottomSheet(
      context: context,
      builder: (context) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Select Test Image',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: TestImages.all.map((assetPath) {
                return GestureDetector(
                  onTap: () async {
                    Navigator.pop(context);
                    final file = await TestImages.getFileFromAsset(assetPath);
                    onImageSelected(file);
                  },
                  child: Column(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.asset(
                          assetPath,
                          width: 100,
                          height: 100,
                          fit: BoxFit.cover,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        TestImages.getName(assetPath),
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton({
    required BuildContext context,
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isEnabled = true,
    bool isActive = false,
  }) {
    return InkWell(
      onTap: isEnabled ? onTap : null,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
        decoration: BoxDecoration(
          color: isActive
              ? Theme.of(context).colorScheme.primaryContainer
              : null,
          border: Border.all(
            color: isActive
                ? Theme.of(context).colorScheme.primary
                : Theme.of(context).colorScheme.outline.withValues(alpha: 0.5),
          ),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 20,
              color: isEnabled
                  ? (isActive
                        ? Theme.of(context).colorScheme.primary
                        : Theme.of(context).colorScheme.primary)
                  : Theme.of(context).colorScheme.outline,
            ),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                label,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: isEnabled
                      ? (isActive
                            ? Theme.of(context).colorScheme.primary
                            : null)
                      : Theme.of(context).colorScheme.outline,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
