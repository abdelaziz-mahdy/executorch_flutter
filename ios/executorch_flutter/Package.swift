// swift-tools-version: 5.9
// ExecuTorch Flutter is a dart:ffi plugin. Its native library is built and
// bundled through Flutter's native assets system (see hook/build.dart), not
// compiled here. This Swift package exists so the plugin advertises Swift
// Package Manager support; it ships only a placeholder target and the privacy
// manifest. CocoaPods support is retained via executorch_flutter.podspec.
import PackageDescription

let package = Package(
    name: "executorch_flutter",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "executorch-flutter", targets: ["executorch_flutter"])
    ],
    targets: [
        .target(
            name: "executorch_flutter",
            resources: [
                .process("PrivacyInfo.xcprivacy")
            ]
        )
    ]
)
