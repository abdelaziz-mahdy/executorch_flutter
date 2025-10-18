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
Supports ARM64 devices and simulators with optimized performance for mobile ML workloads.
                       DESC
  s.homepage         = 'https://github.com/pytorch/executorch'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'PyTorch Team' => 'executorch@pytorch.org' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'

  # iOS deployment target aligned with ExecuTorch 1.0.0 requirements
  s.platform = :ios, '17.0'
  s.swift_version = '5.9'

  # ExecuTorch dependency is handled via Package.swift when Swift Package Manager is enabled
  # The Package.swift file contains the ExecuTorch 1.0.0 dependency configuration (swiftpm-1.0.0 branch)

  # Required iOS system frameworks for ExecuTorch operation
  s.frameworks = [
    'Accelerate',          # BLAS/LAPACK operations
    'CoreML',              # CoreML backend support
    'MetalPerformanceShaders', # MPS backend support
    'Foundation',          # Base iOS framework
    'UIKit'               # iOS UI framework
  ]

  # Additional system libraries required by ExecuTorch
  s.libraries = ['c++', 'resolv']

  # Basic build configuration for Swift package integration
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'VALID_ARCHS' => 'arm64 x86_64',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'OTHER_LDFLAGS' => '-ObjC'
  }

  # Additional build settings for release optimization
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }

  # Note: ExecuTorch dependency will be resolved via CocoaPods from the Swift package
end