/**
 * ExecuTorch Model Manager - Core iOS ExecuTorch Integration
 *
 * This class provides the core ExecuTorch integration for the iOS platform,
 * implementing model lifecycle management, inference execution, and tensor data conversion.
 * It integrates with ExecuTorch iOS frameworks using the Module/Tensor/Value API pattern.
 *
 * Features:
 * - ExecuTorch Module integration with proper error handling
 * - Thread-safe model management
 * - Efficient tensor data conversion between Flutter and ExecuTorch formats
 * - User-controlled memory management and resource cleanup
 * - Performance monitoring and comprehensive error handling
 * - Model metadata extraction and validation
 * - Memory mapping support for large models
 *
 * Integration Pattern:
 * - Uses ExecuTorchModule for model loading and inference
 * - ExecuTorchTensor/ExecuTorchValue API for input/output data handling
 * - Async/await for non-blocking operations
 * - User-controlled model lifecycle
 *
 * Requirements:
 * - iOS 13.0+ for ExecuTorch framework support
 * - ExecuTorch.framework linked and embedded
 * - Swift 5.9+ for async/await support
 */
import Foundation

#if os(iOS)
import Flutter
#elseif os(macOS)
import FlutterMacOS
#endif

// ExecuTorch only supports arm64 architecture on macOS
#if os(macOS)
  #if arch(arm64)
    import ExecuTorch
  #else
    #error("ExecuTorch only supports arm64 (Apple Silicon) on macOS. Intel Macs (x86_64) are NOT supported. Add to macos/Runner/Configs/Release.xcconfig: ARCHS = arm64 and ONLY_ACTIVE_ARCH = YES")
  #endif
#else
  import ExecuTorch
#endif

/**
 * Actor-based model manager for thread-safe ExecuTorch operations
 */
actor ExecutorchModelManager {

    internal static let TAG = "ExecutorchModelManager"
    private static let MODEL_ID_PREFIX = "executorch_model_"

    // Model storage
    private var loadedModels: [String: LoadedModel] = [:]
    private var modelCounter: Int = 0

    // Debug logging control
    private var isDebugLoggingEnabled = false

    /**
     * Represents a loaded ExecuTorch model with metadata
     */
    private struct LoadedModel {
        let module: Module
        let filePath: String
        let loadTime: TimeInterval = Date().timeIntervalSince1970
    }

    init() {
        // Note: log() cannot be used here because init is synchronous
        // and we want to log initialization regardless of debug setting
        print("[\(Self.TAG)] ExecutorchModelManager initialized")
    }

    deinit {
        // Note: deinit is synchronous, so we use print directly
        if isDebugLoggingEnabled {
            print("[\(Self.TAG)] ExecutorchModelManager deinitialized")
        }
    }

    // MARK: - Public API

    /**
     * Load an ExecuTorch model from a file path
     * Throws ExecutorchError on failure
     */
    func load(filePath: String) async throws -> ModelLoadResult {
        log("Loading ExecuTorch model from: \(filePath)")

        // Validate file exists and is readable
        guard FileManager.default.fileExists(atPath: filePath) else {
            logError("File not found: \(filePath)")
            throw ExecutorchError.fileNotFound(filePath)
        }

        guard FileManager.default.isReadableFile(atPath: filePath) else {
            logError("File not readable: \(filePath)")
            throw ExecutorchError.modelLoadFailed(filePath, nil)
        }

        // Generate unique model ID
        let modelId = generateModelId()

        do {
            // Load model using ExecuTorch Module API
            log("Creating Module with file path")
            let module = Module(filePath: filePath)

            // Load the forward method (most common case)
            log("Loading 'forward' method")
            try module.load("forward")

            // Verify the module is loaded
            guard module.isLoaded("forward") else {
                logError("Module.isLoaded('forward') returned false for: \(filePath)")
                throw ExecutorchError.modelLoadFailed(filePath, nil)
            }

            // Store loaded model
            let loadedModel = LoadedModel(
                module: module,
                filePath: filePath
            )

            loadedModels[modelId] = loadedModel

            log("Successfully loaded model: \(modelId) from \(filePath)")

            return ModelLoadResult(modelId: modelId)

        } catch {
            // Clean up on failure
            loadedModels.removeValue(forKey: modelId)

            logError("Failed to load ExecuTorch module: \(filePath), error: \(error)")
            throw ExecutorchError.modelLoadFailed(filePath, error)
        }
    }

    /**
     * Run forward pass (inference) on a loaded model
     * Throws ExecutorchError on failure
     * Returns output tensors directly
     */
    func forward(modelId: String, inputs: [TensorData?]) async throws -> [TensorData?] {
        guard let loadedModel = loadedModels[modelId] else {
            logError("Model not found: \(modelId)")
            throw ExecutorchError.modelNotFound(modelId)
        }

        log("Running forward pass on model: \(modelId)")

        // Convert Flutter tensors to ExecuTorch Values
        let validInputs = inputs.compactMap { $0 }
        let inputValues = try convertTensorsToValues(validInputs)

        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            // Execute inference using ExecuTorch Module.forward()
            let outputs = try loadedModel.module.forward(inputValues)

            let endTime = CFAbsoluteTimeGetCurrent()
            let executionTimeMs = (endTime - startTime) * 1000

            // Convert outputs back to Flutter tensors
            let outputTensors = try convertValuesToTensors(outputs)

            log("Forward pass completed in \(executionTimeMs)ms for model: \(modelId)")

            return outputTensors

        } catch {
            logError("Forward pass failed for model: \(modelId), error: \(error)")
            throw ExecutorchError.inferenceFailed(modelId, error)
        }
    }

    /**
     * Dispose a loaded model and free its resources
     * Throws ExecutorchError.modelNotFound if model not found
     */
    func dispose(modelId: String) async throws {
        let loadedModel = loadedModels.removeValue(forKey: modelId)

        guard loadedModel != nil else {
            logError("Cannot dispose - model not found: \(modelId)")
            throw ExecutorchError.modelNotFound(modelId)
        }

        // Note: ExecuTorchModule cleanup handled by ARC
        log("Disposed model: \(modelId)")
    }

    /**
     * Get list of loaded model IDs
     */
    func getLoadedModels() throws -> [String?] {
        return Array(loadedModels.keys)
    }

    /**
     * Enable or disable debug logging
     */
    func setDebugLogging(enabled: Bool) throws {
        isDebugLoggingEnabled = enabled
        log("Debug logging \(enabled ? "enabled" : "disabled")")
    }

    // MARK: - Private Helper Methods

    /**
     * Log a debug message (only if debug logging is enabled)
     */
    private func log(_ message: String) {
        if isDebugLoggingEnabled {
            print("[\(Self.TAG)] \(message)")
        }
    }

    /**
     * Log an error message (always logged regardless of debug setting)
     */
    private func logError(_ message: String) {
        NSLog("[\(Self.TAG)] ERROR: \(message)")
    }

    private func generateModelId() -> String {
        modelCounter += 1
        return "\(Self.MODEL_ID_PREFIX)\(UUID().uuidString.prefix(8))"
    }

    private func convertTensorsToValues(_ tensors: [TensorData]) throws -> [Value] {
        return try tensors.map { tensorData in
            let shape = tensorData.shape.compactMap { $0 }.map { Int($0) }

            switch tensorData.dataType {
            case .float32:
                // Convert bytes back to float array
                let bytes = tensorData.data.data
                let floatCount = bytes.count / 4
                var floats = [Float](repeating: 0, count: floatCount)
                _ = floats.withUnsafeMutableBytes { floatBytes in
                    bytes.copyBytes(to: floatBytes)
                }

                let tensor = Tensor(floats, shape: shape)
                return Value(tensor)

            case .int32:
                // Convert bytes back to int32 array
                let bytes = tensorData.data.data
                let intCount = bytes.count / 4
                var ints = [Int32](repeating: 0, count: intCount)
                _ = ints.withUnsafeMutableBytes { intBytes in
                    bytes.copyBytes(to: intBytes)
                }

                let tensor = Tensor(ints, shape: shape)
                return Value(tensor)

            case .int8, .uint8:
                // Use bytes directly
                let bytes = tensorData.data.data
                let byteArray = Array(bytes)
                let tensor = Tensor(byteArray, shape: shape)
                return Value(tensor)
            }
        }
    }

    private func convertValuesToTensors(_ values: [Value]) throws -> [TensorData] {
        return try values.enumerated().map { (index, value) in
            // Try to get tensor as Float first (most common), then Int32, then Byte
            if let tensor: Tensor<Float> = value.tensor() {
                let shape = tensor.shape.map { Int64($0) }
                let floatArray = tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) -> [Float] in
                    Array(buffer)
                }
                let data = Data(bytes: floatArray, count: floatArray.count * MemoryLayout<Float>.size)
                return TensorData(
                    shape: shape,
                    dataType: .float32,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } else if let tensor: Tensor<Int32> = value.tensor() {
                let shape = tensor.shape.map { Int64($0) }
                let intArray = tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) -> [Int32] in
                    Array(buffer)
                }
                let data = Data(bytes: intArray, count: intArray.count * MemoryLayout<Int32>.size)
                return TensorData(
                    shape: shape,
                    dataType: .int32,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } else if let tensor: Tensor<UInt8> = value.tensor() {
                let shape = tensor.shape.map { Int64($0) }
                let byteArray = tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt8>) -> [UInt8] in
                    Array(buffer)
                }
                let data = Data(byteArray)
                return TensorData(
                    shape: shape,
                    dataType: .uint8,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } else {
                throw ExecutorchError.inferenceFailed("Output \(index) has unsupported tensor type", nil)
            }
        }
    }

}