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
#if os(iOS)
import Flutter
import UIKit
#elseif os(macOS)
import FlutterMacOS
import AppKit
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
 * Custom log sink for ExecuTorch debug logging
 */
class ExecutorchLogSink: LogSink {
    func log(level: LogLevel, timestamp: TimeInterval, filename: String, line: UInt, message: String) {
        let levelStr: String
        switch level {
        case .debug:
            levelStr = "DEBUG"
        case .info:
            levelStr = "INFO"
        case .error:
            levelStr = "ERROR"
        case .fatal:
            levelStr = "FATAL"
        @unknown default:
            levelStr = "UNKNOWN"
        }

        let fileName = (filename as NSString).lastPathComponent
        print("[\(levelStr)] ExecuTorch [\(fileName):\(line)] \(message)")
    }
}

/**
 * Main plugin class that implements the Flutter plugin interface and Pigeon-generated APIs
 */
public class ExecutorchFlutterPlugin: NSObject, FlutterPlugin, ExecutorchHostApi {

    private static let TAG = "ExecutorchFlutter"

    // Model management
    private var modelManager: ExecutorchModelManager!

    // ExecuTorch logging
    private var logSink: ExecutorchLogSink?

    // MARK: - Flutter Plugin Registration

    public static func register(with registrar: FlutterPluginRegistrar) {
        let plugin = ExecutorchFlutterPlugin()

        // Set up Pigeon-generated method channel
        // iOS uses messenger() method, macOS uses messenger property
        #if os(iOS)
        ExecutorchHostApiSetup.setUp(binaryMessenger: registrar.messenger(), api: plugin)
        #elseif os(macOS)
        ExecutorchHostApiSetup.setUp(binaryMessenger: registrar.messenger, api: plugin)
        #endif

        print("[\(Self.TAG)] ExecutorchFlutterPlugin registered")
    }

    public override init() {
        super.init()

        // Initialize the model manager
        self.modelManager = ExecutorchModelManager()

        print("[\(Self.TAG)] ExecutorchFlutterPlugin initialized")
    }

    deinit {
        // Note: Models are cleaned up when the model manager is deallocated
        print("[\(Self.TAG)] ExecutorchFlutterPlugin deinitialized")
    }

    // MARK: - Pigeon ExecutorchHostApi Implementation

    public func load(filePath: String, completion: @escaping (Result<ModelLoadResult, Error>) -> Void) {
        print("[\(Self.TAG)] Loading model from: \(filePath)")

        Task {
            do {
                let result = try await self.modelManager.load(filePath: filePath)
                print("[\(Self.TAG)] Model loaded successfully: \(result.modelId)")
                completion(.success(result))
            } catch let executorchError as ExecutorchError {
                print("[\(Self.TAG)] Failed to load model: \(filePath), error: \(executorchError)")
                let pigeonError = PigeonError(
                    code: executorchError.errorCode,
                    message: executorchError.localizedDescription,
                    details: nil
                )
                completion(.failure(pigeonError))
            } catch {
                print("[\(Self.TAG)] Failed to load model: \(filePath), error: \(error)")
                let pigeonError = PigeonError(
                    code: "UNKNOWN_ERROR",
                    message: error.localizedDescription,
                    details: nil
                )
                completion(.failure(pigeonError))
            }
        }
    }

    public func forward(modelId: String, inputs: [TensorData?], completion: @escaping (Result<[TensorData?], Error>) -> Void) {
        print("[\(Self.TAG)] Running inference for model: \(modelId)")
        let startTime = CFAbsoluteTimeGetCurrent()

        Task {
            do {
                let outputs = try await self.modelManager.forward(modelId: modelId, inputs: inputs)

                let endTime = CFAbsoluteTimeGetCurrent()
                let executionTime = (endTime - startTime) * 1000 // Convert to milliseconds

                print("[\(Self.TAG)] Inference completed in \(executionTime)ms for model: \(modelId)")

                completion(.success(outputs))
            } catch let executorchError as ExecutorchError {
                print("[\(Self.TAG)] Inference failed for model: \(modelId), error: \(executorchError)")
                let pigeonError = PigeonError(
                    code: executorchError.errorCode,
                    message: executorchError.localizedDescription,
                    details: nil
                )
                completion(.failure(pigeonError))
            } catch {
                print("[\(Self.TAG)] Inference failed for model: \(modelId), error: \(error)")
                let pigeonError = PigeonError(
                    code: "UNKNOWN_ERROR",
                    message: error.localizedDescription,
                    details: nil
                )
                completion(.failure(pigeonError))
            }
        }
    }

    public func dispose(modelId: String) throws {
        print("[\(Self.TAG)] Disposing model: \(modelId)")

        // Use nonisolated(unsafe) to allow mutation from Task
        // This is safe because we use semaphore to synchronize
        nonisolated(unsafe) var thrownError: Error? = nil

        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await modelManager.dispose(modelId: modelId)
            } catch {
                thrownError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let error = thrownError {
            throw error
        }
        print("[\(Self.TAG)] Model disposed successfully: \(modelId)")
    }

    public func getLoadedModels() throws -> [String?] {
        // Use nonisolated(unsafe) to allow mutation from Task
        // This is safe because we use semaphore to synchronize
        nonisolated(unsafe) var result: [String?] = []
        nonisolated(unsafe) var thrownError: Error? = nil

        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                result = try await modelManager.getLoadedModels()
            } catch {
                thrownError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let error = thrownError {
            throw error
        }
        print("[\(Self.TAG)] Currently loaded models: \(result.count)")
        return result
    }

    public func setDebugLogging(enabled: Bool) throws {
        #if DEBUG
        if enabled {
            if logSink == nil {
                logSink = ExecutorchLogSink()
                Log.shared.add(sink: logSink!)
                print("[\(Self.TAG)] ✅ ExecuTorch debug logging ENABLED")
            }
        } else {
            if let sink = logSink {
                Log.shared.remove(sink: sink)
                logSink = nil
                print("[\(Self.TAG)] ❌ ExecuTorch debug logging DISABLED")
            }
        }
        #else
        print("[\(Self.TAG)] ⚠️  Debug logging only available in DEBUG builds")
        #endif
    }

    // MARK: - Private Helper Methods

    /**
     * Execute async code synchronously for Pigeon compatibility
     * Pigeon doesn't support async methods yet, so we need to bridge async/await to sync
     */
    private func executeSync<T>(_ operation: @escaping () async throws -> T) throws -> T {
        // Use nonisolated(unsafe) to allow mutation from Task
        // This is safe because we use semaphore to synchronize
        nonisolated(unsafe) var result: Result<T, Error>?
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

    var errorCode: String {
        switch self {
        case .modelNotFound: return "MODEL_NOT_FOUND"
        case .modelLoadFailed: return "MODEL_LOAD_FAILED"
        case .inferenceFailed: return "INFERENCE_FAILED"
        case .validationError: return "VALIDATION_ERROR"
        case .internalError: return "INTERNAL_ERROR"
        case .fileNotFound: return "FILE_NOT_FOUND"
        case .invalidModelFormat: return "INVALID_MODEL_FORMAT"
        case .memoryError: return "MEMORY_ERROR"
        }
    }

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
