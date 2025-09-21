/**
 * ExecuTorch Flutter Plugin - iOS Platform Implementation
 *
 * This class implements the iOS-specific platform interface for the ExecuTorch Flutter plugin.
 * It uses Pigeon-generated interfaces for type-safe communication with the Dart side and integrates
 * with ExecuTorch iOS frameworks using the Module/Tensor/Value API pattern.
 *
 * Features:
 * - Type-safe method channel communication via Pigeon
 * - ExecuTorch iOS framework integration with Module.load() API pattern
 * - Thread-safe model management with lifecycle handling
 * - Proper memory management with ARC compliance
 * - Performance monitoring and error handling
 * - Memory mapping support for large models
 *
 * Architecture:
 * - ExecutorchFlutterPlugin: Main plugin entry point and Pigeon interface implementation
 * - ExecutorchModelManager: Core model lifecycle and inference management
 * - Background queue execution for non-blocking model operations
 * - Proper iOS lifecycle awareness and memory management
 *
 * Requirements:
 * - iOS 13.0+ (ExecuTorch framework requirement)
 * - ExecuTorch.framework linked and embedded
 * - Swift 5.9+ for async/await support
 */
import Flutter
import UIKit

/**
 * Main plugin class that implements the Flutter plugin interface and Pigeon-generated APIs
 */
public class ExecutorchFlutterPlugin: NSObject, FlutterPlugin, ExecutorchHostApi {

    private static let TAG = "ExecutorchFlutter"

    // Model management
    private var modelManager: ExecutorchModelManager!
    private var lifecycleManager: ExecutorchLifecycleManager!

    // Background queue for model operations
    private let modelQueue = DispatchQueue(label: "com.zcreations.executorch_flutter.models",
                                         qos: .default)

    // MARK: - Flutter Plugin Registration

    public static func register(with registrar: FlutterPluginRegistrar) {
        let plugin = ExecutorchFlutterPlugin()

        // Set up Pigeon-generated method channel
        ExecutorchHostApiSetup.setUp(binaryMessenger: registrar.messenger(), api: plugin)

        print("[\(TAG)] ExecutorchFlutterPlugin registered")
    }

    public override init() {
        super.init()

        // Initialize the lifecycle manager
        self.lifecycleManager = ExecutorchLifecycleManager.shared

        // Initialize the model manager
        self.modelManager = ExecutorchModelManager(queue: modelQueue)

        // Register model manager with lifecycle manager
        self.lifecycleManager.registerModelManager(modelManager)

        print("[\(TAG)] ExecutorchFlutterPlugin initialized")
    }

    deinit {
        // Unregister from lifecycle manager
        lifecycleManager.unregisterModelManager(modelManager)

        // Clean up all models
        Task {
            await modelManager.disposeAllModels()
        }
        print("[\(TAG)] ExecutorchFlutterPlugin deinitialized")
    }

    // MARK: - Pigeon ExecutorchHostApi Implementation

    public func loadModel(filePath: String, completion: @escaping (Result<ModelLoadResult, Error>) -> Void) {
        modelQueue.async { [weak self] in
            guard let self = self else {
                completion(.failure(ExecutorchError.internalError("Plugin instance deallocated")))
                return
            }

            print("[\(Self.TAG)] Loading model from: \(filePath)")

            Task {
                do {
                    let result = try await self.modelManager.loadModel(filePath: filePath)
                    print("[\(Self.TAG)] Model loaded successfully: \(result.modelId)")
                    completion(.success(result))
                } catch {
                    print("[\(Self.TAG)] Failed to load model: \(filePath), error: \(error)")
                    let errorResult = ModelLoadResult(
                        modelId: "",
                        state: ModelState.error,
                        metadata: nil,
                        errorMessage: "Failed to load model: \(error.localizedDescription)"
                    )
                    completion(.success(errorResult))
                }
            }
        }
    }

    public func runInference(request: InferenceRequest, completion: @escaping (Result<InferenceResult, Error>) -> Void) {
        modelQueue.async { [weak self] in
            guard let self = self else {
                completion(.failure(ExecutorchError.internalError("Plugin instance deallocated")))
                return
            }

            print("[\(Self.TAG)] Running inference for model: \(request.modelId)")
            let startTime = CFAbsoluteTimeGetCurrent()

            Task {
                do {
                    let result = try await self.modelManager.runInference(request: request)

                    let endTime = CFAbsoluteTimeGetCurrent()
                    let executionTime = (endTime - startTime) * 1000 // Convert to milliseconds

                    print("[\(Self.TAG)] Inference completed in \(executionTime)ms for model: \(request.modelId)")

                    // Override execution time with measured value if not set
                    let finalResult: InferenceResult
                    if result.executionTimeMs == 0.0 {
                        finalResult = InferenceResult(
                            status: result.status,
                            executionTimeMs: executionTime,
                            requestId: result.requestId,
                            outputs: result.outputs,
                            errorMessage: result.errorMessage,
                            metadata: result.metadata
                        )
                    } else {
                        finalResult = result
                    }
                    completion(.success(finalResult))
                } catch {
                    print("[\(Self.TAG)] Inference failed for model: \(request.modelId), error: \(error)")
                    let errorResult = InferenceResult(
                        status: InferenceStatus.error,
                        executionTimeMs: 0.0,
                        requestId: request.requestId,
                        outputs: nil,
                        errorMessage: "Inference failed: \(error.localizedDescription)",
                        metadata: nil
                    )
                    completion(.success(errorResult))
                }
            }
        }
    }

    public func getModelMetadata(modelId: String) throws -> ModelMetadata? {
        do {
            print("[\(Self.TAG)] Getting metadata for model: \(modelId)")
            return try modelManager.getModelMetadata(modelId: modelId)
        } catch {
            print("[\(Self.TAG)] Failed to get metadata for model: \(modelId), error: \(error)")
            return nil
        }
    }

    public func disposeModel(modelId: String) throws {
        do {
            print("[\(Self.TAG)] Disposing model: \(modelId)")
            try modelManager.disposeModel(modelId: modelId)
            print("[\(Self.TAG)] Model disposed successfully: \(modelId)")
        } catch {
            print("[\(Self.TAG)] Failed to dispose model: \(modelId), error: \(error)")
            throw error
        }
    }

    public func getLoadedModels() throws -> [String?] {
        do {
            let models = try modelManager.getLoadedModelIds()
            print("[\(Self.TAG)] Currently loaded models: \(models.count)")
            return models.map { $0 as String? }
        } catch {
            print("[\(Self.TAG)] Failed to get loaded models, error: \(error)")
            return []
        }
    }

    public func getModelState(modelId: String) throws -> ModelState {
        do {
            let state = try modelManager.getModelState(modelId: modelId)
            print("[\(Self.TAG)] Model \(modelId) state: \(state)")
            return state
        } catch {
            print("[\(Self.TAG)] Failed to get state for model: \(modelId), error: \(error)")
            return ModelState.error
        }
    }

    // MARK: - Private Helper Methods

    /**
     * Execute async code synchronously for Pigeon compatibility
     * Pigeon doesn't support async methods yet, so we need to bridge async/await to sync
     */
    private func executeSync<T>(_ operation: @escaping () async throws -> T) throws -> T {
        var result: Result<T, Error>?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                let value = try await operation()
                result = .success(value)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()

        switch result! {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        }
    }
}

// MARK: - Error Types

enum ExecutorchError: Error, LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String, Error?)
    case inferenceFailed(String, Error?)
    case validationError(String)
    case internalError(String)
    case fileNotFound(String)
    case invalidModelFormat(String)
    case memoryError(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let modelId):
            return "Model not found: \(modelId)"
        case .modelLoadFailed(let path, let error):
            return "Failed to load model from \(path): \(error?.localizedDescription ?? "Unknown error")"
        case .inferenceFailed(let modelId, let error):
            return "Inference failed for model \(modelId): \(error?.localizedDescription ?? "Unknown error")"
        case .validationError(let message):
            return "Validation error: \(message)"
        case .internalError(let message):
            return "Internal error: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidModelFormat(let path):
            return "Invalid model format: \(path)"
        case .memoryError(let message):
            return "Memory error: \(message)"
        }
    }
}
