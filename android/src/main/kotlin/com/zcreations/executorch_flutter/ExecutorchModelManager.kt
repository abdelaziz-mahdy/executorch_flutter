/**
 * ExecuTorch Model Manager - Core Android ExecuTorch Integration
 *
 * This class provides the core ExecuTorch integration for the Android platform,
 * implementing model lifecycle management, inference execution, and tensor data conversion.
 * It integrates with the latest ExecuTorch Android AAR 0.7.0+ using the Module.load() API pattern.
 *
 * Features:
 * - ExecuTorch Module.load() integration with proper exception handling
 * - Thread-safe model management with concurrent access support
 * - Efficient tensor data conversion between Flutter and ExecuTorch formats
 * - Memory management with proper resource cleanup
 * - Performance monitoring and comprehensive error handling
 * - Model metadata extraction and validation
 *
 * Integration Pattern:
 * - Uses org.pytorch.executorch.Module for model loading and inference
 * - EValue/Tensor API for input/output data handling
 * - Proper Android context awareness and lifecycle management
 * - Background thread execution for non-blocking operations
 */
package com.zcreations.executorch_flutter

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.collections.HashMap

// Import Pigeon generated API
import com.zcreations.executorch_flutter.generated.*

/**
 * Manages ExecuTorch model lifecycle and inference operations
 */
class ExecutorchModelManager(
    private val context: Context,
    private val scope: CoroutineScope
) {
    companion object {
        private const val TAG = "ExecutorchModelManager"
        private const val MAX_CONCURRENT_MODELS = 5
        private const val MODEL_ID_PREFIX = "executorch_model_"
    }

    // Model storage and state management
    private val loadedModels = ConcurrentHashMap<String, LoadedModel>()
    private val modelStates = ConcurrentHashMap<String, ModelState>()
    private val modelCounter = java.util.concurrent.atomic.AtomicInteger(0)

    /**
     * Represents a loaded ExecuTorch model with metadata
     */
    private data class LoadedModel(
        val module: Module,
        val metadata: ModelMetadata,
        val filePath: String,
        val loadTime: Long = System.currentTimeMillis()
    )

    /**
     * Load an ExecuTorch model from a file path
     */
    suspend fun loadModel(filePath: String): ModelLoadResult = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Loading ExecuTorch model from: $filePath")

            // Validate file exists and is readable
            val file = File(filePath)
            if (!file.exists()) {
                throw ModelLoadException("Model file not found: $filePath")
            }

            if (!file.canRead()) {
                throw ModelLoadException("Model file not readable: $filePath")
            }

            // Check model count limit
            if (loadedModels.size >= MAX_CONCURRENT_MODELS) {
                throw ModelLoadException("Maximum number of concurrent models ($MAX_CONCURRENT_MODELS) reached")
            }

            // Generate unique model ID
            val modelId = generateModelId()

            // Update state to loading
            modelStates[modelId] = ModelState.LOADING

            try {
                // Load model using ExecuTorch Module.load() API
                Log.d(TAG, "Loading ExecuTorch module with Module.load()")
                val module = Module.load(filePath)

                // Extract model metadata
                val metadata = extractModelMetadata(module, filePath)

                // Store loaded model
                val loadedModel = LoadedModel(
                    module = module,
                    metadata = metadata,
                    filePath = filePath
                )

                loadedModels[modelId] = loadedModel
                modelStates[modelId] = ModelState.READY

                Log.d(TAG, "Successfully loaded model: $modelId from $filePath")

                ModelLoadResult(
                    modelId = modelId,
                    state = ModelState.READY,
                    metadata = metadata,
                    errorMessage = null
                )

            } catch (e: Exception) {
                // Clean up on failure
                modelStates[modelId] = ModelState.ERROR
                loadedModels.remove(modelId)

                Log.e(TAG, "Failed to load ExecuTorch module: $filePath", e)
                throw ModelLoadException("ExecuTorch module loading failed: ${e.message}", e)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Model loading failed: $filePath", e)

            ModelLoadResult(
                modelId = "",
                state = ModelState.ERROR,
                metadata = null,
                errorMessage = "Failed to load model: ${e.message}"
            )
        }
    }

    /**
     * Run inference on a loaded model
     */
    suspend fun runInference(request: InferenceRequest): InferenceResult = withContext(Dispatchers.Default) {
        try {
            val modelId = request.modelId
            val loadedModel = loadedModels[modelId]
                ?: throw ModelNotFoundException("Model not found: $modelId")

            Log.d(TAG, "Running inference on model: $modelId")

            // Filter out null inputs and validate
            val validInputs = request.inputs.filterNotNull()
            validateInputTensors(validInputs, loadedModel.metadata)

            // Convert Flutter tensors to ExecuTorch EValues
            val inputEValues = convertTensorsToEValues(validInputs)

            val startTime = System.nanoTime()

            // Execute inference using ExecuTorch Module.forward()
            val outputs = loadedModel.module.forward(*inputEValues.toTypedArray())

            val endTime = System.nanoTime()
            val executionTimeMs = (endTime - startTime) / 1_000_000.0

            // Convert outputs back to Flutter tensors
            val outputTensors = convertEValuesToTensors(outputs)

            Log.d(TAG, "Inference completed in ${executionTimeMs}ms for model: $modelId")

            InferenceResult(
                status = InferenceStatus.SUCCESS,
                executionTimeMs = executionTimeMs,
                requestId = request.requestId,
                outputs = outputTensors,
                errorMessage = null,
                metadata = createInferenceMetadata(loadedModel) as Map<String?, Any?>?
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed for model: ${request.modelId}", e)

            InferenceResult(
                status = InferenceStatus.ERROR,
                executionTimeMs = 0.0,
                requestId = request.requestId,
                outputs = null,
                errorMessage = "Inference failed: ${e.message}",
                metadata = null
            )
        }
    }

    /**
     * Get metadata for a loaded model
     */
    fun getModelMetadata(modelId: String): ModelMetadata? {
        return loadedModels[modelId]?.metadata
    }

    /**
     * Dispose a loaded model and free resources
     */
    suspend fun disposeModel(modelId: String) = withContext(Dispatchers.IO) {
        try {
            val loadedModel = loadedModels.remove(modelId)
            modelStates[modelId] = ModelState.DISPOSED

            if (loadedModel != null) {
                // Note: ExecuTorch Module doesn't have explicit dispose method
                // Rely on garbage collection for cleanup
                Log.d(TAG, "Disposed model: $modelId")
            } else {
                Log.w(TAG, "Attempted to dispose unknown model: $modelId")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to dispose model: $modelId", e)
            throw e
        }
    }

    /**
     * Dispose all loaded models
     */
    suspend fun disposeAllModels() = withContext(Dispatchers.IO) {
        val modelIds = loadedModels.keys.toList()
        for (modelId in modelIds) {
            try {
                disposeModel(modelId)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to dispose model during cleanup: $modelId", e)
            }
        }
    }

    /**
     * Get the current state of a model
     */
    fun getModelState(modelId: String): ModelState {
        return modelStates[modelId] ?: ModelState.ERROR
    }

    /**
     * Get list of loaded model IDs
     */
    fun getLoadedModelIds(): List<String> {
        return loadedModels.keys.toList()
    }

    // Private helper methods

    private fun generateModelId(): String {
        return MODEL_ID_PREFIX + UUID.randomUUID().toString().replace("-", "").substring(0, 8)
    }

    private fun extractModelMetadata(module: Module, filePath: String): ModelMetadata {
        // Note: ExecuTorch Module doesn't provide introspection APIs
        // We'll create basic metadata from file information
        val file = File(filePath)
        val modelName = file.nameWithoutExtension

        // Create placeholder tensor specs - in a real implementation,
        // these would come from model introspection or external metadata
        val inputSpecs = listOf(
            TensorSpec(
                name = "input",
                shape = listOf(1L, 3L, 224L, 224L), // Common image input shape
                dataType = TensorType.FLOAT32,
                optional = false,
                validRange = null
            )
        )

        val outputSpecs = listOf(
            TensorSpec(
                name = "output",
                shape = listOf(1L, 1000L), // Common classification output
                dataType = TensorType.FLOAT32,
                optional = false,
                validRange = null
            )
        )

        // Estimate memory usage based on file size
        val estimatedMemoryMB = (file.length() / (1024 * 1024)).toLong().coerceAtLeast(1L)

        val properties = mapOf(
            "file_path" to filePath,
            "file_size_bytes" to file.length(),
            "load_time" to System.currentTimeMillis(),
            "backend" to "executorch_android"
        )

        return ModelMetadata(
            modelName = modelName,
            version = "1.0.0",
            inputSpecs = inputSpecs,
            outputSpecs = outputSpecs,
            estimatedMemoryMB = estimatedMemoryMB,
            properties = properties as Map<String?, Any?>?
        )
    }

    private fun validateInputTensors(inputs: List<TensorData>, metadata: ModelMetadata) {
        if (inputs.size != metadata.inputSpecs.size) {
            throw ValidationException(
                "Input count mismatch: expected ${metadata.inputSpecs.size}, got ${inputs.size}"
            )
        }

        // Additional validation can be added here
        for (i in inputs.indices) {
            val input = inputs[i]
            val spec = metadata.inputSpecs[i]

            if (spec != null && input.dataType != spec.dataType) {
                throw ValidationException(
                    "Input $i data type mismatch: expected ${spec.dataType}, got ${input.dataType}"
                )
            }
        }
    }

    private fun convertTensorsToEValues(tensors: List<TensorData>): List<EValue> {
        return tensors.map { tensorData ->
            val tensor = when (tensorData.dataType) {
                TensorType.FLOAT32 -> {
                    val buffer = ByteBuffer.wrap(tensorData.data)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                    val floatArray = FloatArray(buffer.remaining())
                    buffer.get(floatArray)

                    Tensor.fromBlob(
                        floatArray,
                        tensorData.shape.filterNotNull().map { it.toLong() }.toLongArray()
                    )
                }

                TensorType.INT32 -> {
                    val buffer = ByteBuffer.wrap(tensorData.data)
                        .order(ByteOrder.nativeOrder())
                        .asIntBuffer()
                    val intArray = IntArray(buffer.remaining())
                    buffer.get(intArray)

                    Tensor.fromBlob(
                        intArray,
                        tensorData.shape.filterNotNull().map { it.toLong() }.toLongArray()
                    )
                }

                TensorType.INT8, TensorType.UINT8 -> {
                    Tensor.fromBlob(
                        tensorData.data,
                        tensorData.shape.filterNotNull().map { it.toLong() }.toLongArray()
                    )
                }
            }

            EValue.from(tensor)
        }
    }

    private fun convertEValuesToTensors(eValues: Array<EValue>): List<TensorData> {
        return eValues.mapIndexed { index, eValue ->
            val tensor = eValue.toTensor()
            val shape = tensor.shape().map { it.toInt() }

            // Determine data type and convert accordingly
            val (dataType, data) = when {
                tensor.dataAsFloatArray != null -> {
                    val floatArray = tensor.dataAsFloatArray
                    val buffer = ByteBuffer.allocate(floatArray.size * 4)
                        .order(ByteOrder.nativeOrder())
                    buffer.asFloatBuffer().put(floatArray)
                    TensorType.FLOAT32 to buffer.array()
                }

                tensor.dataAsIntArray != null -> {
                    val intArray = tensor.dataAsIntArray
                    val buffer = ByteBuffer.allocate(intArray.size * 4)
                        .order(ByteOrder.nativeOrder())
                    buffer.asIntBuffer().put(intArray)
                    TensorType.INT32 to buffer.array()
                }

                else -> {
                    // Default to byte array
                    TensorType.UINT8 to tensor.dataAsByteArray
                }
            }

            TensorData(
                shape = shape.map { it.toLong() },
                dataType = dataType,
                data = data,
                name = "output_$index"
            )
        }
    }

    private fun createInferenceMetadata(loadedModel: LoadedModel): Map<String, Any> {
        return mapOf(
            "model_id" to loadedModel.hashCode().toString(),
            "backend" to "executorch_android",
            "timestamp" to System.currentTimeMillis(),
            "model_file" to loadedModel.filePath
        )
    }
}