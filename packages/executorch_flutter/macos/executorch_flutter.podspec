#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint executorch_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'executorch_flutter'
  s.version          = '0.1.1'
  s.summary          = 'ExecuTorch ML inference for Flutter using dart:ffi.'
  s.description      = <<-DESC
ExecuTorch ML inference for Flutter using dart:ffi.
Native code is built via native assets - this podspec is a placeholder.
                       DESC
  s.homepage         = 'https://github.com/abdelaziz-mahdy/executorch_flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Abdelaziz Mahdy' => 'abdelaziz.h.mahdy@gmail.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.platform         = :osx, '11.0'
  s.swift_version    = '5.0'

  # Native code is built via native assets hook, not through CocoaPods
  # This is just a placeholder to satisfy Flutter's plugin discovery
end
