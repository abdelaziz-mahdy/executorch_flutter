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
 * - Uses Module for model loading and inference (ExecuTorch 0.7.0+ API)
 * - Tensor/Value API for input/output data handling
 * - Proper iOS memory mapping and lifecycle management
 * - Background queue execution for non-blocking operations
 *
 * Requirements:
 * - iOS 13.0+ for ExecuTorch framework support
 * - ExecuTorch.framework linked and embedded
 * - Swift 5.9+ for async/await support
 */
import Foundation
import ExecuTorch

#if os(iOS)
import Flutter
#elseif os(macOS)
import FlutterMacOS
#endif

/**
 * Actor-based model manager for thread-safe ExecuTorch operations
 */
actor ExecutorchModelManager {

    internal static let TAG = "ExecutorchModelManager"
    private static let MAX_CONCURRENT_MODELS = 5
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
        let metadata: ModelMetadata
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

        // Check file size
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: filePath)
            if let fileSize = attributes[.size] as? Int64 {
                let fileSizeMB = Double(fileSize) / (1024 * 1024)
                print("[\(Self.TAG)] Model file size: \(String(format: "%.2f", fileSizeMB)) MB")

                // Warn if file is very small (might be corrupted)
                if fileSize < 1024 {
                    print("[\(Self.TAG)] WARNING: Model file is very small (\(fileSize) bytes), might be corrupted")
                }
            }
        } catch {
            print("[\(Self.TAG)] Warning: Could not get file size: \(error)")
        }

        // Check model count limit
        guard loadedModels.count < Self.MAX_CONCURRENT_MODELS else {
            throw ExecutorchError.memoryError("Maximum number of concurrent models (\(Self.MAX_CONCURRENT_MODELS)) reached")
        }

        // Generate unique model ID
        let modelId = generateModelId()

        // Update state to loading
        modelStates[modelId] = ModelState.loading

        do {
            // Load model using ExecuTorch Module API
            // Note: Module just stores the path at creation. The actual .pte file will be loaded
            // when forward() is called for the first time, so the file must exist until then.
            print("[\(Self.TAG)] Creating Module with file path")
            let module = Module(filePath: filePath)

            // Verify file size right before we consider the module ready
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: filePath)
                if let fileSize = attributes[.size] as? Int64 {
                    let fileSizeMB = Double(fileSize) / (1024 * 1024)
                    print("[\(Self.TAG)] File size: \(String(format: "%.2f", fileSizeMB)) MB")

                    // Read first few bytes to verify it's a valid .pte file
                    if let fileHandle = FileHandle(forReadingAtPath: filePath) {
                        let headerData = fileHandle.readData(ofLength: 16)
                        let headerHex = headerData.map { String(format: "%02x", $0) }.joined(separator: " ")
                        print("[\(Self.TAG)] File header (first 16 bytes): \(headerHex)")
                        fileHandle.closeFile()
                    }
                }
            } catch {
                print("[\(Self.TAG)] Warning: Could not verify file: \(error)")
            }

            print("[\(Self.TAG)] Module created successfully (actual loading will happen on first forward() call)")

            // Extract model metadata
            let metadata = try extractModelMetadata(from: module, filePath: filePath)

            // Store loaded model
            let loadedModel = LoadedModel(
                module: module,
                metadata: metadata,
                filePath: filePath
            )

            loadedModels[modelId] = loadedModel
            modelStates[modelId] = ModelState.ready

            print("[\(Self.TAG)] Successfully loaded model: \(modelId) from \(filePath)")

            return ModelLoadResult(
                modelId: modelId,
                state: ModelState.ready,
                metadata: metadata,
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

        // Check model file still exists and size before inference
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: loadedModel.filePath)
            if let fileSize = attributes[.size] as? Int64 {
                let fileSizeMB = Double(fileSize) / (1024 * 1024)
                print("[\(Self.TAG)] Model file size before inference: \(String(format: "%.2f", fileSizeMB)) MB")
            }
        } catch {
            print("[\(Self.TAG)] Warning: Could not verify model file before inference: \(error)")
        }

        // Unwrap optional TensorData array
        let inputs = request.inputs.compactMap { $0 }

        // Validate input tensors
        try validateInputTensors(inputs, against: loadedModel.metadata)

        // Convert Flutter tensors to ExecuTorch Values
        let inputValues = try convertTensorsToValues(inputs)

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
                metadata: createInferenceMetadata(from: loadedModel)
            )

        } catch {
            print("[\(Self.TAG)] Inference failed for model: \(request.modelId), error: \(error)")
            throw ExecutorchError.inferenceFailed(request.modelId, error)
        }
    }

    /**
     * Get metadata for a loaded model
     */
    func getModelMetadata(modelId: String) throws -> ModelMetadata? {
        return loadedModels[modelId]?.metadata
    }

    /**
     * Dispose a loaded model and free resources
     */
    func disposeModel(modelId: String) async throws {
        let loadedModel = loadedModels.removeValue(forKey: modelId)
        modelStates[modelId] = ModelState.disposed

        if loadedModel != nil {
            // Note: Module doesn't have explicit dispose method
            // Rely on ARC for cleanup
            print("[\(Self.TAG)] Disposed model: \(modelId)")
        } else {
            print("[\(Self.TAG)] Attempted to dispose unknown model: \(modelId)")
        }
    }

    /**
     * Dispose all loaded models
     */
    func disposeAllModels() async {
        let modelIds = Array(loadedModels.keys)
        for modelId in modelIds {
            do {
                try await disposeModel(modelId: modelId)
            } catch {
                print("[\(Self.TAG)] Failed to dispose model during cleanup: \(modelId), error: \(error)")
            }
        }
    }

    /**
     * Get the current state of a model
     */
    func getModelState(modelId: String) throws -> ModelState {
        return modelStates[modelId] ?? ModelState.error
    }

    /**
     * Get list of loaded model IDs
     */
    func getLoadedModelIds() throws -> [String] {
        return Array(loadedModels.keys)
    }

    // MARK: - Private Helper Methods

    private func generateModelId() -> String {
        modelCounter += 1
        return "\(Self.MODEL_ID_PREFIX)\(UUID().uuidString.prefix(8))"
    }

    private func extractModelMetadata(from module: Module, filePath: String) throws -> ModelMetadata {
        // Note: Module doesn't provide introspection APIs
        // We'll create basic metadata from file information
        let fileURL = URL(fileURLWithPath: filePath)
        let modelName = fileURL.deletingPathExtension().lastPathComponent

        // Create placeholder tensor specs - in a real implementation,
        // these would come from model introspection or external metadata
        let inputSpecs = [
            TensorSpec(
                name: "input",
                shape: [1, 3, 224, 224], // Common image input shape
                dataType: TensorType.float32,
                optional: false,
                validRange: nil
            )
        ]

        let outputSpecs = [
            TensorSpec(
                name: "output",
                shape: [1, 1000], // Common classification output
                dataType: TensorType.float32,
                optional: false,
                validRange: nil
            )
        ]

        // Estimate memory usage based on file size
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: filePath)
        let fileSize = fileAttributes[.size] as? Int64 ?? 0
        let estimatedMemoryMB = Int(max(fileSize / (1024 * 1024), 1))

        let properties: [String: Any] = [
            "file_path": filePath,
            "file_size_bytes": fileSize,
            "load_time": Date().timeIntervalSince1970,
            "backend": "executorch_ios"
        ]

        return ModelMetadata(
            modelName: modelName,
            version: "1.0.0",
            inputSpecs: inputSpecs,
            outputSpecs: outputSpecs,
            estimatedMemoryMB: Int64(estimatedMemoryMB),
            properties: properties
        )
    }

    private func validateInputTensors(_ inputs: [TensorData], against metadata: ModelMetadata) throws {
        guard inputs.count == metadata.inputSpecs.count else {
            throw ExecutorchError.validationError(
                "Input count mismatch: expected \(metadata.inputSpecs.count), got \(inputs.count)"
            )
        }

        // Additional validation can be added here
        for (index, input) in inputs.enumerated() {
            guard let spec = metadata.inputSpecs[index] else {
                throw ExecutorchError.validationError("Input spec at index \(index) is nil")
            }

            guard input.dataType == spec.dataType else {
                throw ExecutorchError.validationError(
                    "Input \(index) data type mismatch: expected \(spec.dataType), got \(input.dataType)"
                )
            }
        }
    }

    private func convertTensorsToValues(_ tensors: [TensorData]) throws -> [Value] {
        return try tensors.map { tensorData in
            let value: Value

            switch tensorData.dataType {
            case .float32:
                // Convert bytes back to float array
                let data = tensorData.data.data
                let floatCount = data.count / 4
                var floats = [Float](repeating: 0, count: floatCount)
                _ = floats.withUnsafeMutableBytes { floatBytes in
                    data.copyBytes(to: floatBytes)
                }

                let tensor = Tensor<Float>(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: floats),
                    shape: tensorData.shape.compactMap { $0 }.map { Int($0) }
                )
                value = Value(tensor)

            case .int32:
                // Convert bytes back to int32 array
                let data = tensorData.data.data
                let intCount = data.count / 4
                var ints = [Int32](repeating: 0, count: intCount)
                _ = ints.withUnsafeMutableBytes { intBytes in
                    data.copyBytes(to: intBytes)
                }

                let tensor = Tensor<Int32>(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ints),
                    shape: tensorData.shape.compactMap { $0 }.map { Int($0) }
                )
                value = Value(tensor)

            case .int8:
                // Use bytes directly for int8
                var data = tensorData.data.data
                let tensor = try data.withUnsafeMutableBytes { bytes -> Tensor<Int8> in
                    Tensor<Int8>(
                        bytesNoCopy: bytes.baseAddress!,
                        shape: tensorData.shape.compactMap { $0 }.map { Int($0) }
                    )
                }
                value = Value(tensor)

            case .uint8:
                // Use bytes directly for uint8
                var data = tensorData.data.data
                let tensor = try data.withUnsafeMutableBytes { bytes -> Tensor<UInt8> in
                    Tensor<UInt8>(
                        bytesNoCopy: bytes.baseAddress!,
                        shape: tensorData.shape.compactMap { $0 }.map { Int($0) }
                    )
                }
                value = Value(tensor)
            }

            return value
        }
    }

    private func convertValuesToTensors(_ values: [Value]) throws -> [TensorData] {
        return try values.enumerated().map { (index, value) in
            // Try to extract tensor as different types using ExecuTorch 1.0.0 API
            // First try Float (most common for ML models)
            do {
                let floatTensor = try Tensor<Float>(value)
                let shape = floatTensor.shape.map { Int64($0) }
                let scalars = floatTensor.scalars()
                let data = Data(bytes: scalars, count: scalars.count * MemoryLayout<Float>.stride)

                return TensorData(
                    shape: shape,
                    dataType: TensorType.float32,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } catch {}

            // Try Int32
            do {
                let int32Tensor = try Tensor<Int32>(value)
                let shape = int32Tensor.shape.map { Int64($0) }
                let scalars = int32Tensor.scalars()
                let data = Data(bytes: scalars, count: scalars.count * MemoryLayout<Int32>.stride)

                return TensorData(
                    shape: shape,
                    dataType: TensorType.int32,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } catch {}

            // Try UInt8
            do {
                let uint8Tensor = try Tensor<UInt8>(value)
                let shape = uint8Tensor.shape.map { Int64($0) }
                let scalars = uint8Tensor.scalars()
                let data = Data(bytes: scalars, count: scalars.count)

                return TensorData(
                    shape: shape,
                    dataType: TensorType.uint8,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } catch {}

            // Try Int8
            do {
                let int8Tensor = try Tensor<Int8>(value)
                let shape = int8Tensor.shape.map { Int64($0) }
                let scalars = int8Tensor.scalars()
                let data = Data(bytes: scalars, count: scalars.count)

                return TensorData(
                    shape: shape,
                    dataType: TensorType.int8,
                    data: FlutterStandardTypedData(bytes: data),
                    name: "output_\(index)"
                )
            } catch {}

            throw ExecutorchError.inferenceFailed("Output \(index) has unsupported tensor type", nil)
        }
    }

    private func createInferenceMetadata(from loadedModel: LoadedModel) -> [String: Any] {
        return [
            "model_id": loadedModel.filePath.hashValue,
            "backend": "executorch_macos",
            "timestamp": Date().timeIntervalSince1970,
            "model_file": loadedModel.filePath
        ]
    }
}