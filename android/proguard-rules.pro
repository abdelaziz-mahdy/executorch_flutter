# ExecuTorch Flutter Plugin ProGuard Rules
# These rules prevent ProGuard from removing or obfuscating classes needed at runtime

# Keep all ExecuTorch classes
# ExecuTorch uses reflection and JNI to access these classes
-keep class org.pytorch.executorch.** { *; }
-dontwarn org.pytorch.executorch.**

# Keep fbjni (Facebook JNI) classes
# Required for Java-to-native bridging
-keep class com.facebook.jni.** { *; }
-dontwarn com.facebook.jni.**

# Keep SoLoader classes
# Required for loading native libraries
-keep class com.facebook.soloader.** { *; }
-dontwarn com.facebook.soloader.**
