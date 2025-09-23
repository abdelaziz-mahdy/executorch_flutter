#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Generate Test Models for ExecuTorch Flutter Plugin
#
# This script generates sample models for testing the Flutter plugin
# across different platforms and backends, including image, text, and audio models.

import os
import sys
import json

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        print("✓ PyTorch is available")

        try:
            import torchvision
            print("✓ TorchVision is available")
        except ImportError:
            print("⚠️  TorchVision not available, using basic models only")

        try:
            import torchaudio
            print("✓ TorchAudio is available")
        except ImportError:
            print("⚠️  TorchAudio not available, using basic audio models only")

        # Check if our custom exporter is available
        try:
            from executorch_exporter import ExecuTorchExporter
            print("✓ ExecuTorch exporter is available")
            return True
        except ImportError:
            print("⚠️  ExecuTorch exporter not available, creating mock models instead")
            return False

    except ImportError as e:
        print(f"✗ Missing PyTorch: {e}")
        print("\nPlease install PyTorch:")
        print("pip install torch")
        return False

def generate_classification_models():
    """Generate image classification models for testing."""
    print("📱 Generating image classification models...")

    from executorch_exporter import ExecuTorchExporter, ExportConfig
    import torchvision.models as models
    import torch

    # MobileNetV3 Small - Good for mobile testing
    model = models.mobilenet_v3_small(weights='DEFAULT').eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exporter = ExecuTorchExporter()

    # Generate models for all three backends
    configs = [
        # CoreML for iOS optimization
        ExportConfig(
            model_name="mobilenet_v3_small_coreml",
            backends=["coreml"],
            output_dir="../example/assets/models",
            quantize=False
        ),
        # XNNPACK for CPU optimization (both platforms)
        ExportConfig(
            model_name="mobilenet_v3_small_xnnpack",
            backends=["xnnpack"],
            output_dir="../example/assets/models",
            quantize=False
        ),
        # MPS for GPU optimization (macOS/iOS)
        ExportConfig(
            model_name="mobilenet_v3_small_mps",
            backends=["mps"],
            output_dir="../example/assets/models",
            quantize=False
        )
    ]

    # Export models for all backends
    successful = 0
    for config in configs:
        try:
            results = exporter.export_model(model, sample_inputs, config)
            successful += sum(1 for r in results if r.success)
        except Exception as e:
            print(f"⚠️  Failed to generate {config.model_name}: {e}")

    print(f"✓ Generated {successful} image classification models")

def generate_text_classification_models():
    """Generate text classification models for testing."""
    print("📝 Generating text classification models...")

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Simple BERT-like text classifier
        class SimpleTextClassifier(nn.Module):
            def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=64, num_classes=3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.classifier = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.1)

            def forward(self, input_ids, attention_mask=None):
                # input_ids: [batch_size, seq_len]
                embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]

                if attention_mask is not None:
                    # Apply attention mask by zeroing out padded positions
                    embedded = embedded * attention_mask.unsqueeze(-1).float()

                lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]

                # Use the last hidden state for classification
                output = self.classifier(self.dropout(hidden.squeeze(0)))  # [batch_size, num_classes]
                return output

        # Create multiple text models for comprehensive testing
        exporter = ExecuTorchExporter()

        # 1. Sentiment analysis model (3 classes: negative, neutral, positive)
        sentiment_model = SimpleTextClassifier(vocab_size=1000, num_classes=3).eval()
        seq_len = 128
        sample_input_ids = torch.randint(0, 1000, (1, seq_len))
        sample_attention_mask = torch.ones(1, seq_len)
        sample_inputs = (sample_input_ids, sample_attention_mask)

        # Generate models for all backends
        text_models = [
            (sentiment_model, "sentiment_analysis", 3, False),
            (SimpleTextClassifier(vocab_size=1000, num_classes=5).eval(), "topic_classification", 5, False),
            (SimpleTextClassifier(vocab_size=1000, num_classes=10).eval(), "language_detection", 10, True)
        ]

        successful = 0
        for model_instance, model_name, num_classes, quantize in text_models:
            configs = [
                ExportConfig(
                    model_name=f"{model_name}_xnnpack",
                    backends=["xnnpack"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                ),
                ExportConfig(
                    model_name=f"{model_name}_mps",
                    backends=["mps"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                ),
                ExportConfig(
                    model_name=model_name,
                    backends=["coreml"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                )
            ]

            for config in configs:
                try:
                    results = exporter.export_model(model_instance, sample_inputs, config)
                    successful += sum(1 for r in results if r.success)
                except Exception as e:
                    print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} text classification models")

        # Generate vocabulary and label files for the tokenizers
        _generate_demo_vocabulary()
        _generate_text_class_labels()

    except Exception as e:
        print(f"⚠️  Text classification model generation failed: {e}")

def generate_audio_classification_models():
    """Generate audio classification models for testing."""
    print("🎵 Generating audio classification models...")

    try:
        import torch
        import torch.nn as nn
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Simple CNN for audio classification
        class SimpleAudioClassifier(nn.Module):
            def __init__(self, input_features=80, num_classes=10):
                super().__init__()
                # Expecting mel-spectrogram input: [batch, time_frames, mel_bands]
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, num_classes)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                # x shape: [batch, time_frames, mel_bands]
                # Add channel dimension: [batch, 1, time_frames, mel_bands]
                x = x.unsqueeze(1)

                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)  # [batch, 64, 1, 1]
                x = x.view(x.size(0), -1)  # [batch, 64]
                x = self.dropout(x)
                x = self.classifier(x)  # [batch, num_classes]
                return x

        # Create multiple audio models for comprehensive testing
        exporter = ExecuTorchExporter()

        # 1. Environmental sound classifier (10 classes)
        env_model = SimpleAudioClassifier(input_features=80, num_classes=10).eval()
        sample_inputs_env = (torch.randn(1, 100, 80),)  # mel-spectrogram format

        # Generate models for all backends
        audio_models = [
            (env_model, "environmental_sound_classifier", False),
            (SimpleAudioClassifier(input_features=80, num_classes=7).eval(), "speech_emotion_recognition", False),
            (SimpleAudioClassifier(input_features=80, num_classes=8).eval(), "music_genre_classifier", True),
            (SimpleAudioClassifier(input_features=80, num_classes=2).eval(), "voice_activity_detector", True)
        ]

        successful = 0
        for model_instance, model_name, quantize in audio_models:
            configs = [
                ExportConfig(
                    model_name=f"{model_name}_xnnpack",
                    backends=["xnnpack"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                ),
                ExportConfig(
                    model_name=f"{model_name}_mps",
                    backends=["mps"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                ),
                ExportConfig(
                    model_name=model_name,
                    backends=["coreml"],
                    output_dir="../example/assets/models",
                    quantize=quantize
                )
            ]

            for config in configs:
                try:
                    results = exporter.export_model(model_instance, sample_inputs_env, config)
                    successful += sum(1 for r in results if r.success)
                except Exception as e:
                    print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} audio classification models")

        # Generate class labels files for all audio models
        _generate_audio_class_labels()
        _generate_audio_emotion_labels()
        _generate_music_genre_labels()
        _generate_vad_labels()

    except Exception as e:
        print(f"⚠️  Audio classification model generation failed: {e}")

def generate_simple_demo_model():
    """Generate a very simple model for basic testing."""
    print("🔧 Generating simple demo model...")

    try:
        from executorch_exporter import ExecuTorchExporter, ExportConfig
        import torch
        import torch.nn as nn

        # Simple linear model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().eval()
        sample_inputs = (torch.randn(1, 10),)

        exporter = ExecuTorchExporter()
        # Generate demo models for all backends
        configs = [
            ExportConfig(
                model_name="simple_demo_xnnpack",
                backends=["xnnpack"],
                output_dir="../example/assets/models",
                quantize=False
            ),
            ExportConfig(
                model_name="simple_demo_mps",
                backends=["mps"],
                output_dir="../example/assets/models",
                quantize=False
            ),
            ExportConfig(
                model_name="simple_demo_coreml",
                backends=["coreml"],
                output_dir="../example/assets/models",
                quantize=False
            )
        ]

        successful = 0
        for config in configs:
            try:
                results = exporter.export_model(model, sample_inputs, config)
                successful += sum(1 for r in results if r.success)
            except Exception as e:
                print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} demo models")
    except Exception as e:
        print(f"⚠️  Simple demo model generation failed: {e}")

def _generate_demo_vocabulary():
    """Generate a demo vocabulary file for text processing."""
    vocabulary = {
        "<pad>": 0, "<unk>": 1, "the": 2, "a": 3, "an": 4, "and": 5, "or": 6, "but": 7,
        "in": 8, "on": 9, "at": 10, "to": 11, "for": 12, "of": 13, "with": 14, "by": 15,
        "this": 16, "that": 17, "these": 18, "those": 19, "i": 20, "you": 21, "he": 22,
        "she": 23, "it": 24, "we": 25, "they": 26, "me": 27, "him": 28, "her": 29,
        "us": 30, "them": 31, "my": 32, "your": 33, "his": 34, "its": 35, "our": 36,
        "their": 37, "is": 38, "am": 39, "are": 40, "was": 41, "were": 42, "be": 43,
        "been": 44, "being": 45, "have": 46, "has": 47, "had": 48, "do": 49, "does": 50,
        "did": 51, "will": 52, "would": 53, "could": 54, "should": 55, "may": 56,
        "might": 57, "can": 58,
        # Positive sentiment words
        "love": 59, "like": 60, "enjoy": 61, "amazing": 62, "fantastic": 63, "great": 64,
        "excellent": 65, "wonderful": 66, "awesome": 67, "perfect": 68, "happy": 69,
        "excited": 70, "pleased": 71, "satisfied": 72, "delighted": 73, "thrilled": 74,
        "good": 75, "best": 76, "beautiful": 77, "nice": 78,
        # Negative sentiment words
        "hate": 79, "dislike": 80, "terrible": 81, "awful": 82, "horrible": 83, "bad": 84,
        "worst": 85, "disappointed": 86, "angry": 87, "frustrated": 88, "sad": 89,
        "upset": 90, "annoyed": 91, "disgusted": 92, "furious": 93, "miserable": 94,
        "depressed": 95, "stressed": 96, "worried": 97,
        # Neutral words
        "okay": 98, "fine": 99, "normal": 100, "average": 101, "typical": 102,
        "regular": 103, "standard": 104, "usual": 105, "common": 106, "ordinary": 107,
    }

    # Save vocabulary as JSON
    vocab_path = "../example/assets/models/demo_vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocabulary, f, indent=2)
    print(f"✓ Generated demo vocabulary: {vocab_path}")

def _generate_text_class_labels():
    """Generate text classification class labels."""
    # Topic classification labels
    topic_labels = [
        "business", "technology", "sports", "politics", "entertainment"
    ]

    # Language detection labels
    language_labels = [
        "english", "spanish", "french", "german", "italian",
        "portuguese", "russian", "chinese", "japanese", "arabic"
    ]

    # Sentiment labels (already in vocabulary function, but for completeness)
    sentiment_labels = ["negative", "neutral", "positive"]

    # Save all text classification labels
    labels_data = {
        "topic_classification": topic_labels,
        "language_detection": language_labels,
        "sentiment_analysis": sentiment_labels
    }

    labels_path = "../example/assets/models/text_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_data, f, indent=2)
    print(f"✓ Generated text class labels: {labels_path}")

def _generate_audio_class_labels():
    """Generate environmental sound class labels file."""
    class_labels = [
        "silence", "speech", "music", "traffic", "nature",
        "machinery", "alarm", "animal", "footsteps", "applause"
    ]

    # Save class labels as JSON
    labels_path = "../example/assets/models/audio_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"✓ Generated environmental sound class labels: {labels_path}")

def _generate_audio_emotion_labels():
    """Generate speech emotion recognition labels."""
    emotion_labels = [
        "neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"
    ]

    labels_path = "../example/assets/models/emotion_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(emotion_labels, f, indent=2)
    print(f"✓ Generated emotion class labels: {labels_path}")

def _generate_music_genre_labels():
    """Generate music genre classification labels."""
    genre_labels = [
        "rock", "pop", "jazz", "classical", "hip_hop", "country", "electronic", "blues"
    ]

    labels_path = "../example/assets/models/music_genre_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(genre_labels, f, indent=2)
    print(f"✓ Generated music genre labels: {labels_path}")

def _generate_vad_labels():
    """Generate voice activity detection labels."""
    vad_labels = ["no_speech", "speech"]

    labels_path = "../example/assets/models/vad_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(vad_labels, f, indent=2)
    print(f"✓ Generated VAD labels: {labels_path}")

def generate_imagenet_labels():
    """Generate ImageNet class labels file."""
    print("🏷️  Generating ImageNet class labels...")

    try:
        import urllib.request

        # Download the official ImageNet class labels from PyTorch Hub
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"Downloading ImageNet labels from: {url}")

        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
            imagenet_labels = content.strip().split('\n')

        # Save to both python directory and example assets
        labels_paths = [
            "imagenet_classes.txt",  # Python directory
            "../example/assets/models/imagenet_classes.txt"  # Example assets
        ]

        for labels_path in labels_paths:
            with open(labels_path, 'w') as f:
                for label in imagenet_labels:
                    f.write(label + '\n')
            print(f"✓ Generated ImageNet labels: {labels_path}")

    except Exception as e:
        print(f"❌ Failed to download ImageNet labels: {e}")
        raise e  # No fallbacks allowed

def generate_object_detection_models():
    """Generate object detection models for real-time camera processing."""
    print("🎯 Generating object detection models...")

    try:
        import torch
        import torch.nn as nn
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Simple object detection model (YOLO-like structure)
        class SimpleObjectDetector(nn.Module):
            def __init__(self, num_classes=80, input_size=320):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((input_size//32, input_size//32)),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((input_size//64, input_size//64)),
                )

                # Detection head: [objectness, x, y, w, h, class_probs...]
                self.detection_head = nn.Conv2d(64, (5 + num_classes), 1)

            def forward(self, x):
                features = self.backbone(x)
                detections = self.detection_head(features)
                return detections

        # Create and export object detection model
        model = SimpleObjectDetector(num_classes=20, input_size=320).eval()  # 20 common object classes
        sample_inputs = (torch.randn(1, 3, 320, 320),)

        exporter = ExecuTorchExporter()
        # Generate object detection models for all backends
        configs = [
            ExportConfig(
                model_name="simple_object_detector_xnnpack",
                backends=["xnnpack"],
                output_dir="../example/assets/models",
                quantize=False
            ),
            ExportConfig(
                model_name="simple_object_detector_mps",
                backends=["mps"],
                output_dir="../example/assets/models",
                quantize=False
            ),
            ExportConfig(
                model_name="simple_object_detector_coreml",
                backends=["coreml"],
                output_dir="../example/assets/models",
                quantize=False
            )
        ]

        successful = 0
        for config in configs:
            try:
                results = exporter.export_model(model, sample_inputs, config)
                successful += sum(1 for r in results if r.success)
            except Exception as e:
                print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} object detection models")

        # Generate COCO class labels subset
        _generate_detection_class_labels()

    except Exception as e:
        print(f"⚠️  Object detection model generation failed: {e}")

def generate_realtime_classification_models():
    """Generate lightweight models optimized for real-time camera processing."""
    print("⚡ Generating real-time classification models...")

    try:
        import torch
        import torch.nn as nn
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Ultra-lightweight classifier for real-time processing
        class RealtimeClassifier(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                # Optimized for speed over accuracy
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((56, 56)),
                    nn.Conv2d(16, 32, 5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((14, 14)),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Linear(64, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        # Create and export real-time classifier
        model = RealtimeClassifier(num_classes=10).eval()
        sample_inputs = (torch.randn(1, 3, 224, 224),)

        exporter = ExecuTorchExporter()
        # Generate realtime models for all backends
        configs = [
            ExportConfig(
                model_name="realtime_classifier_xnnpack",
                backends=["xnnpack"],
                output_dir="../example/assets/models",
                quantize=True
            ),
            ExportConfig(
                model_name="realtime_classifier_mps",
                backends=["mps"],
                output_dir="../example/assets/models",
                quantize=True
            ),
            ExportConfig(
                model_name="realtime_classifier_coreml",
                backends=["coreml"],
                output_dir="../example/assets/models",
                quantize=True
            )
        ]

        successful = 0
        for config in configs:
            try:
                results = exporter.export_model(model, sample_inputs, config)
                successful += sum(1 for r in results if r.success)
            except Exception as e:
                print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} real-time classification models")

        # Generate simple class labels for demo
        _generate_realtime_class_labels()

    except Exception as e:
        print(f"⚠️  Real-time classification model generation failed: {e}")

def generate_face_detection_models():
    """Generate face detection models for camera demos."""
    print("👤 Generating face detection models...")

    try:
        import torch
        import torch.nn as nn
        from executorch_exporter import ExecuTorchExporter, ExportConfig

        # Simple face detection model
        class SimpleFaceDetector(nn.Module):
            def __init__(self):
                super().__init__()
                # Lightweight architecture for face detection
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                )

                # Face detection head: [objectness, x, y, w, h]
                self.detection_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 128),
                    nn.ReLU(),
                    nn.Linear(128, 5)  # [confidence, x, y, w, h]
                )

            def forward(self, x):
                features = self.backbone(x)
                detection = self.detection_head(features)
                return detection

        # Create and export face detection model
        model = SimpleFaceDetector().eval()
        sample_inputs = (torch.randn(1, 3, 224, 224),)

        exporter = ExecuTorchExporter()
        # Generate face detection models for all backends
        configs = [
            ExportConfig(
                model_name="simple_face_detector_xnnpack",
                backends=["xnnpack"],
                output_dir="../example/assets/models",
                quantize=True
            ),
            ExportConfig(
                model_name="simple_face_detector_mps",
                backends=["mps"],
                output_dir="../example/assets/models",
                quantize=True
            ),
            ExportConfig(
                model_name="simple_face_detector_coreml",
                backends=["coreml"],
                output_dir="../example/assets/models",
                quantize=True
            )
        ]

        successful = 0
        for config in configs:
            try:
                results = exporter.export_model(model, sample_inputs, config)
                successful += sum(1 for r in results if r.success)
            except Exception as e:
                print(f"⚠️  Failed to generate {config.model_name}: {e}")

        print(f"✓ Generated {successful} face detection models")

    except Exception as e:
        print(f"⚠️  Face detection model generation failed: {e}")

def _generate_detection_class_labels():
    """Generate detection class labels file."""
    # Common object classes for detection demo
    class_labels = [
        "person", "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow"
    ]

    labels_path = "../example/assets/models/detection_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"✓ Generated detection class labels: {labels_path}")

def _generate_realtime_class_labels():
    """Generate real-time classification class labels."""
    class_labels = [
        "person", "animal", "vehicle", "food", "object",
        "building", "nature", "technology", "furniture", "other"
    ]

    labels_path = "../example/assets/models/realtime_class_labels.json"
    with open(labels_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"✓ Generated real-time class labels: {labels_path}")

def generate_models():
    """Generate all test models."""
    print("🚀 ExecuTorch Test Model Generation")
    print("=" * 50)

    # Create output directory
    os.makedirs("../example/assets/models", exist_ok=True)

    try:
        # Generate basic demo model first
        generate_simple_demo_model()

        # Generate image classification models
        generate_classification_models()
        generate_imagenet_labels()

        # Generate text classification models
        generate_text_classification_models()

        # Generate audio classification models
        generate_audio_classification_models()

        # Generate new models for camera and real-time processing
        generate_object_detection_models()
        generate_realtime_classification_models()
        generate_face_detection_models()

        print("\n✅ Model generation completed!")
        print("Models saved to: ../example/assets/models/")
        print("\nGenerated files (all with xnnpack, mps, and coreml variants):")
        print("📱 Image Models:")
        print("  - mobilenet_v3_small_[backend].pte (ImageNet classification)")
        print("  - imagenet_classes.txt")
        print("📝 Text Models:")
        print("  - sentiment_analysis_[backend].pte (3 classes)")
        print("  - topic_classification_[backend].pte (5 topics)")
        print("  - language_detection_[backend].pte (10 languages)")
        print("  - demo_vocabulary.json + text_class_labels.json")
        print("🎵 Audio Models:")
        print("  - environmental_sound_classifier_[backend].pte (10 classes)")
        print("  - speech_emotion_recognition_[backend].pte (7 emotions)")
        print("  - music_genre_classifier_[backend].pte (8 genres)")
        print("  - voice_activity_detector_[backend].pte (speech detection)")
        print("  - Labels: audio_class_labels.json + emotion_class_labels.json + music_genre_labels.json + vad_labels.json")
        print("🎯 Real-time Models:")
        print("  - simple_object_detector_[backend].pte (20 object classes)")
        print("  - realtime_classifier_[backend].pte (10 categories)")
        print("  - simple_face_detector_[backend].pte (face detection)")
        print("  - Labels: detection_class_labels.json + realtime_class_labels.json")
        print("🔧 Demo Models:")
        print("  - simple_demo_[backend].pte (basic linear model)")
        print("\n🚀 Total: 36+ models (12 model types × 3 backends each)!")
        print("\nBackend Performance Guide:")
        print("• xnnpack: CPU optimized, works on all devices")
        print("• mps: Metal Performance Shaders, GPU optimized for Apple devices")
        print("• coreml: iOS/macOS optimized, best performance on Apple devices")
        print("\nNext steps:")
        print("1. Add models to your Flutter app's pubspec.yaml assets section")
        print("2. Implement backend selection UI in your app")
        print("3. Test loading with ExecutorchManager.instance.loadModel()")
        print("4. Use appropriate backend based on device/use case for optimal performance")
        print("5. Benchmark different backends for your specific use cases!")

    except Exception as e:
        print(f"\n❌ Model generation failed: {e}")
        return 1

    return 0

def main():
    if not check_requirements():
        return 1

    return generate_models()

if __name__ == "__main__":
    exit(main())