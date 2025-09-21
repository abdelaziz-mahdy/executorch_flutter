/**
 * ExecuTorch Flutter Plugin - Android Platform Implementation
 *
 * This class implements the Android-specific platform interface for the ExecuTorch Flutter plugin.
 * It uses Pigeon-generated interfaces for type-safe communication with the Dart side and integrates
 * with the ExecuTorch Android AAR library for model loading and inference execution.
 *
 * Features:
 * - Type-safe method channel communication via Pigeon
 * - ExecuTorch AAR 0.7.0+ integration with Module.load() API pattern
 * - Thread-safe model management with lifecycle handling
 * - Proper memory management and resource cleanup
 * - Performance monitoring and error handling
 *
 * Architecture:
 * - ExecutorchFlutterPlugin: Main plugin entry point and Pigeon interface implementation
 * - ExecutorchModelManager: Core model lifecycle and inference management
 * - Background thread execution for non-blocking model operations
 * - Proper Android context and lifecycle awareness
 */
package com.zcreations.executorch_flutter

import android.content.Context
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap

// Import Pigeon generated API
import com.zcreations.executorch_flutter.generated.*

/**
 * Main plugin class that implements the Flutter plugin interface and Pigeon-generated APIs
 */
class ExecutorchFlutterPlugin: FlutterPlugin, ExecutorchHostApi {
    companion object {
        private const val TAG = "ExecutorchFlutter"
    }

    // Plugin lifecycle
    private lateinit var context: Context

    // Model management
    private lateinit var modelManager: ExecutorchModelManager
    private lateinit var lifecycleManager: ExecutorchLifecycleManager

    // Coroutine scope for async operations
    private val pluginScope = CoroutineScope(
        Dispatchers.Default + SupervisorJob() +
        CoroutineName("ExecutorchFlutter")
    )

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        context = flutterPluginBinding.applicationContext

        // Initialize lifecycle manager
        val application = context.applicationContext as android.app.Application
        lifecycleManager = ExecutorchLifecycleManager.getInstance(application)

        // Initialize the model manager
        modelManager = ExecutorchModelManager(context, pluginScope)

        // Register model manager with lifecycle manager
        lifecycleManager.registerModelManager("main", modelManager)

        // Set up Pigeon-generated method channel
        ExecutorchHostApi.setUp(flutterPluginBinding.binaryMessenger, this)

        Log.d(TAG, "ExecutorchFlutterPlugin attached to engine")
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        // Unregister from lifecycle manager
        lifecycleManager.unregisterModelManager("main")

        // Clean up resources
        pluginScope.cancel()

        // Clean up Pigeon method channels
        ExecutorchHostApi.setUp(binding.binaryMessenger, null)

        // Dispose all models
        runBlocking {
            modelManager.disposeAllModels()
        }

        Log.d(TAG, "ExecutorchFlutterPlugin detached from engine")
    }

    // Pigeon ExecutorchHostApi implementation

    override fun loadModel(filePath: String, callback: (Result<ModelLoadResult>) -> Unit) {
        pluginScope.launch {
            try {
                Log.d(TAG, "Loading model from: $filePath")
                val result = modelManager.loadModel(filePath)
                Log.d(TAG, "Model loaded successfully: ${result.modelId}")
                callback(Result.success(result))
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: $filePath", e)
                val errorResult = ModelLoadResult(
                    modelId = "",
                    state = ModelState.ERROR,
                    metadata = null,
                    errorMessage = "Failed to load model: ${e.message}"
                )
                callback(Result.success(errorResult))
            }
        }
    }

    override fun runInference(request: InferenceRequest, callback: (Result<InferenceResult>) -> Unit) {
        pluginScope.launch {
            try {
                Log.d(TAG, "Running inference for model: ${request.modelId}")
                val startTime = System.currentTimeMillis()

                val result = modelManager.runInference(request)

                val endTime = System.currentTimeMillis()
                val executionTime = (endTime - startTime).toDouble()

                Log.d(TAG, "Inference completed in ${executionTime}ms for model: ${request.modelId}")

                // Override execution time with measured value if not set
                val finalResult = if (result.executionTimeMs == 0.0) {
                    InferenceResult(
                        status = result.status,
                        executionTimeMs = executionTime,
                        requestId = result.requestId,
                        outputs = result.outputs,
                        errorMessage = result.errorMessage,
                        metadata = result.metadata
                    )
                } else {
                    result
                }
                callback(Result.success(finalResult))
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed for model: ${request.modelId}", e)
                val errorResult = InferenceResult(
                    status = InferenceStatus.ERROR,
                    executionTimeMs = 0.0,
                    requestId = request.requestId,
                    outputs = null,
                    errorMessage = "Inference failed: ${e.message}",
                    metadata = null
                )
                callback(Result.success(errorResult))
            }
        }
    }

    override fun getModelMetadata(modelId: String): ModelMetadata? {
        return try {
            Log.d(TAG, "Getting metadata for model: $modelId")
            modelManager.getModelMetadata(modelId)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get metadata for model: $modelId", e)
            null
        }
    }

    override fun disposeModel(modelId: String) {
        runBlocking {
            try {
                Log.d(TAG, "Disposing model: $modelId")
                modelManager.disposeModel(modelId)
                Log.d(TAG, "Model disposed successfully: $modelId")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to dispose model: $modelId", e)
            }
        }
    }

    override fun getLoadedModels(): List<String?> {
        return try {
            val models = modelManager.getLoadedModelIds()
            Log.d(TAG, "Currently loaded models: ${models.size}")
            models.map { it as String? }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get loaded models", e)
            emptyList()
        }
    }

    override fun getModelState(modelId: String): ModelState {
        return try {
            val state = modelManager.getModelState(modelId)
            Log.d(TAG, "Model $modelId state: $state")
            state
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get state for model: $modelId", e)
            ModelState.ERROR
        }
    }
}

/**
 * Exception classes for structured error handling
 */
open class ExecutorchException(message: String, cause: Throwable? = null) : Exception(message, cause)

class ModelNotFoundException(modelId: String) : ExecutorchException("Model not found: $modelId")

class ModelLoadException(message: String, cause: Throwable? = null) : ExecutorchException(message, cause)

class InferenceException(message: String, cause: Throwable? = null) : ExecutorchException(message, cause)

class ValidationException(message: String) : ExecutorchException("Validation error: $message")
