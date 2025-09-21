#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint executorch_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'executorch_flutter'
  s.version          = '1.0.0'
  s.summary          = 'A Flutter plugin for ExecuTorch on-device ML inference.'
  s.description      = <<-DESC
A Flutter plugin package that enables on-device machine learning inference using ExecuTorch on iOS platforms.
                       DESC
  s.homepage         = 'https://github.com/pytorch/executorch'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'PyTorch Team' => 'executorch@pytorch.org' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.9'

  # ExecuTorch framework dependencies
  # Note: In a real implementation, these would reference the built frameworks
  # For now, this is a placeholder for the framework integration
  s.vendored_frameworks = [
    'Frameworks/executorch.xcframework',
    'Frameworks/backend_coreml.xcframework',
    'Frameworks/backend_mps.xcframework',
    'Frameworks/backend_xnnpack.xcframework'
  ]

  # Required system frameworks for ExecuTorch iOS integration
  s.frameworks = [
    'Accelerate',
    'CoreML',
    'MetalPerformanceShaders'
  ]

  # Compiler flags for ExecuTorch integration
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_CFLAGS' => '-DEXECUTORCH_BUILD_FLAVOR_OPTIMIZED',
    'GCC_PREPROCESSOR_DEFINITIONS' => 'EXECUTORCH_BUILD_FLAVOR_OPTIMIZED=1'
  }
end