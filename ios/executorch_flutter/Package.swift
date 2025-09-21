// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "executorch_flutter",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "executorch-flutter", targets: ["executorch_flutter"])
    ],
    dependencies: [
        // ExecuTorch Swift Package Manager dependency
        // Using the official swiftpm branch as documented at:
        // https://docs.pytorch.org/executorch/stable/using-executorch-ios.html
        .package(url: "https://github.com/pytorch/executorch.git", branch: "swiftpm-0.7.0")
    ],
    targets: [
        .target(
            name: "executorch_flutter",
            dependencies: [
                // Link ExecuTorch modules as documented
                .product(name: "executorch", package: "executorch"),
                .product(name: "backend_xnnpack", package: "executorch"),
                .product(name: "backend_coreml", package: "executorch"),
                .product(name: "backend_mps", package: "executorch"),
                .product(name: "kernels_optimized", package: "executorch"),
            ],
            path: "Sources/executorch_flutter",
            resources: [
                // Privacy manifest if needed for App Store submission
                // .process("PrivacyInfo.xcprivacy"),
            ],
            cSettings: [
                .headerSearchPath("include")
            ]
        )
    ]
)