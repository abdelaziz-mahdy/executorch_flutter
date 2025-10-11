// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "executorch_flutter",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .library(name: "executorch-flutter", targets: ["executorch_flutter"])
    ],
    dependencies: [
        // ExecuTorch Swift Package Manager dependency
        // Using swiftpm-1.0.0 to match Python nightly build (1.0.0.dev20250916)
        // https://github.com/pytorch/executorch/tree/swiftpm-1.0.0
        .package(url: "https://github.com/pytorch/executorch.git", branch: "swiftpm-1.0.0")
    ],
    targets: [
        .target(
            name: "executorch_flutter",
            dependencies: [
                // Core ExecuTorch runtime
                // Using debug version for development (includes logging support)
                // For release builds, change to "executorch" (without _debug suffix)
                .product(name: "executorch_debug", package: "executorch"),

                // Optional backends - available in both debug and release
                .product(name: "backend_xnnpack", package: "executorch"),  // CPU optimization
                .product(name: "backend_coreml", package: "executorch"),   // CoreML acceleration

                // Note: These backends require iOS 17.0+ / macOS 14.0+
                .product(name: "backend_mps", package: "executorch"),      // Metal Performance Shaders
                .product(name: "kernels_optimized", package: "executorch"), // Optimized kernels
            ],
            path: "Sources/executorch_flutter",
            resources: [
                // Privacy manifest if needed for App Store submission
                // .process("PrivacyInfo.xcprivacy"),
            ],
            cSettings: [
                .headerSearchPath("include")
            ],
            linkerSettings: [
                // Force load all symbols from static libraries to ensure backend registration
                // This is required for ExecuTorch backends to be properly registered
                .unsafeFlags(["-Wl,-all_load"])
            ]
        )
    ]
)
