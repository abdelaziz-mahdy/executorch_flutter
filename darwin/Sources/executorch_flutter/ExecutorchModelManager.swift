/**
 * ExecuTorch Model Manager - Core iOS ExecuTorch Integration
 *
 * This class provides the core ExecuTorch integration for the iOS platform,
 * implementing model lifecycle management, inference execution, and tensor data conversion.
 * It integrates with ExecuTorch iOS frameworks using the Module/Tensor/Value API pattern.
 *
 * Features:
 * - ExecuTorch Module integration with proper error handling
 * - Thread-safe model management with actor-based concurrency
 * - Efficient tensor data conversion between Flutter and ExecuTorch formats
 * - Memory management with ARC compliance and proper resource cleanup
 * - Performance monitoring and comprehensive error handling
 * - Model metadata extraction and validation
 * - Memory mapping support for large models
 *
 * Integration Pattern:
 * - Uses ExecuTorchModule for model loading and inference
 * - ExecuTorchTensor/ExecuTorchValue API for input/output data handling
 * - Proper iOS memory mapping and lifecycle management
 * - Background queue execution for non-blocking operations
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

    // Model storage and state management
    private var loadedModels: [String: LoadedModel] = [:]
    private var modelStates: [String: ModelState] = [:]
    private var modelCounter: Int = 0

    private let queue: DispatchQueue

    /**
     * Represents a loaded ExecuTorch model with metadata
     */
    private struct LoadedModel {
        let module: Module
        let filePath: String
        let loadTime: TimeInterval = Date().timeIntervalSince1970
    }

    init(queue: DispatchQueue) {
        self.queue = queue
        print("[\(Self.TAG)] ExecutorchModelManager initialized")
    }

    deinit {
        print("[\(Self.TAG)] ExecutorchModelManager deinitialized")
    }

    // MARK: - Public API

    /**
     * Load an ExecuTorch model from a file path
     */
    func loadModel(filePath: String) async throws -> ModelLoadResult {
        print("[\(Self.TAG)] Loading ExecuTorch model from: \(filePath)")

        // Validate file exists and is readable
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw ExecutorchError.fileNotFound(filePath)
        }

        guard FileManager.default.isReadableFile(atPath: filePath) else {
            throw ExecutorchError.modelLoadFailed(filePath, nil)
        }

        // Generate unique model ID
        let modelId = generateModelId()

        // Update state to loading
        modelStates[modelId] = ModelState.loading

        do {
            // Load model using ExecuTorch Module API
            print("[\(Self.TAG)] Creating Module with file path")
            let module = Module(filePath: filePath)

            // Load the forward method (most common case)
            print("[\(Self.TAG)] Loading 'forward' method")
            try module.load("forward")

            // Verify the module is loaded
            guard module.isLoaded("forward") else {
                throw ExecutorchError.modelLoadFailed(filePath, nil)
            }

            // Store loaded model
            let loadedModel = LoadedModel(
                module: module,
                filePath: filePath
            )

            loadedModels[modelId] = loadedModel
            modelStates[modelId] = ModelState.ready

            print("[\(Self.TAG)] Successfully loaded model: \(modelId) from \(filePath)")

            return ModelLoadResult(
                modelId: modelId,
                state: ModelState.ready,
                errorMessage: nil
            )

        } catch {
            // Clean up on failure
            modelStates[modelId] = ModelState.error
            loadedModels.removeValue(forKey: modelId)

            print("[\(Self.TAG)] Failed to load ExecuTorch module: \(filePath), error: \(error)")
            throw ExecutorchError.modelLoadFailed(filePath, error)
        }
    }

    /**
     * Run inference on a loaded model
     */
    func runInference(request: InferenceRequest) async throws -> InferenceResult {
        guard let loadedModel = loadedModels[request.modelId] else {
            throw ExecutorchError.modelNotFound(request.modelId)
        }

        print("[\(Self.TAG)] Running inference on model: \(request.modelId)")

        // Convert Flutter tensors to ExecuTorch Values
        let validInputs = request.inputs.compactMap { $0 }
        let inputValues = try convertTensorsToValues(validInputs)

        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            // Execute inference using ExecuTorch Module.forward()
            let outputs = try loadedModel.module.forward(inputValues)

            let endTime = CFAbsoluteTimeGetCurrent()
            let executionTimeMs = (endTime - startTime) * 1000

            // Convert outputs back to Flutter tensors
            let outputTensors = try convertValuesToTensors(outputs)

            print("[\(Self.TAG)] Inference completed in \(executionTimeMs)ms for model: \(request.modelId)")

            return InferenceResult(
                status: InferenceStatus.success,
                executionTimeMs: executionTimeMs,
                requestId: request.requestId,
                outputs: outputTensors,
                errorMessage: nil,
                metadata: nil
            )

        } catch {
            print("[\(Self.TAG)] Inference failed for model: \(request.modelId), error: \(error)")
            throw ExecutorchError.inferenceFailed(request.modelId, error)
        }
    }

    /**
     * Dispose a loaded model and free its resources
     */
    func disposeModel(modelId: String) async throws {
        let loadedModel = loadedModels.removeValue(forKey: modelId)
        modelStates.removeValue(forKey: modelId)

        if loadedModel != nil {
            // Note: ExecuTorchModule cleanup handled by ARC
            print("[\(Self.TAG)] Disposed model: \(modelId)")
        } else {
            print("[\(Self.TAG)] Attempted to dispose unknown model: \(modelId)")
        }
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
        // ExecuTorch doesn't expose a global logging API
        // Logging is controlled at compile time or via environment variables
        print("[\(Self.TAG)] Debug logging setting: \(enabled) (note: may require rebuild)")
    }

    // MARK: - Private Helper Methods

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