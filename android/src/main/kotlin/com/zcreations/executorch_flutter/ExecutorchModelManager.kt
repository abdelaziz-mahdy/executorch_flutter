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
    private val context: Context
) {
    companion object {
        private const val TAG = "ExecutorchModelManager"
        private const val MODEL_ID_PREFIX = "executorch_model_"
    }

    // Model storage
    private val loadedModels = ConcurrentHashMap<String, LoadedModel>()
    private val modelCounter = java.util.concurrent.atomic.AtomicInteger(0)
    private var isDebugLoggingEnabled = false

    /**
     * Represents a loaded ExecuTorch model
     */
    private data class LoadedModel(
        val module: Module,
        val filePath: String,
        val loadTime: Long = System.currentTimeMillis()
    )

    /**
     * Load an ExecuTorch model from a file path
     * Throws ModelLoadException on failure
     */
    suspend fun load(filePath: String): ModelLoadResult = withContext(Dispatchers.IO) {
        if (isDebugLoggingEnabled) {
            Log.d(TAG, "Loading ExecuTorch model from: $filePath")
        }

        // Validate file exists and is readable
        val file = File(filePath)
        if (!file.exists()) {
            throw ModelLoadException("Model file not found: $filePath")
        }

        if (!file.canRead()) {
            throw ModelLoadException("Model file not readable: $filePath")
        }

        // Generate unique model ID
        val modelId = generateModelId()

        try {
            // Load model using ExecuTorch Module.load() API
            if (isDebugLoggingEnabled) {
                Log.d(TAG, "Loading ExecuTorch module with Module.load()")
            }
            val module = Module.load(filePath)

            // Store loaded model
            val loadedModel = LoadedModel(
                module = module,
                filePath = filePath
            )

            loadedModels[modelId] = loadedModel

            if (isDebugLoggingEnabled) {
                Log.d(TAG, "Successfully loaded model: $modelId from $filePath")
            }

            ModelLoadResult(modelId = modelId)

        } catch (e: Exception) {
            // Clean up on failure
            loadedModels.remove(modelId)

            Log.e(TAG, "Failed to load ExecuTorch module: $filePath", e)
            throw ModelLoadException("ExecuTorch module loading failed: ${e.message}", e)
        }
    }

    /**
     * Run forward pass (inference) on a loaded model
     * Returns output tensors directly
     * Throws InferenceException on failure
     */
    suspend fun forward(modelId: String, inputs: List<TensorData?>): List<TensorData?> = withContext(Dispatchers.Default) {
        val loadedModel = loadedModels[modelId]
            ?: throw ModelNotFoundException("Model not found: $modelId")

        if (isDebugLoggingEnabled) {
            Log.d(TAG, "Running inference on model: $modelId")
        }

        try {
            // Filter out null inputs
            val validInputs = inputs.filterNotNull()

            // Convert Flutter tensors to ExecuTorch EValues
            val inputEValues = convertTensorsToEValues(validInputs)

            val startTime = System.nanoTime()

            // Execute inference using ExecuTorch Module.forward()
            val outputs = loadedModel.module.forward(*inputEValues.toTypedArray())

            val endTime = System.nanoTime()
            val executionTimeMs = (endTime - startTime) / 1_000_000.0

            // Convert outputs back to Flutter tensors
            val outputTensors = convertEValuesToTensors(outputs)

            if (isDebugLoggingEnabled) {
                Log.d(TAG, "Inference completed in ${executionTimeMs}ms for model: $modelId")
            }

            outputTensors

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed for model: $modelId", e)
            throw InferenceException("Inference failed: ${e.message}", e)
        }
    }

    /**
     * Dispose a loaded model and free its resources
     * Throws ModelNotFoundException if model not found
     */
    suspend fun dispose(modelId: String) = withContext(Dispatchers.IO) {
        val loadedModel = loadedModels.remove(modelId)
            ?: throw ModelNotFoundException("Model not found: $modelId")

        // Note: ExecuTorch Module cleanup handled by garbage collection
        if (isDebugLoggingEnabled) {
            Log.d(TAG, "Disposed model: $modelId")
        }
    }

    /**
     * Get list of loaded model IDs
     */
    fun getLoadedModels(): List<String?> {
        return loadedModels.keys.toList()
    }

    /**
     * Dispose all loaded models
     */
    suspend fun disposeAllModels() = withContext(Dispatchers.IO) {
        val modelIds = loadedModels.keys.toList()
        modelIds.forEach { modelId ->
            dispose(modelId)
        }
        if (isDebugLoggingEnabled) {
            Log.d(TAG, "Disposed all models (${modelIds.size} total)")
        }
    }

    /**
     * Enable or disable debug logging
     */
    fun setDebugLogging(enabled: Boolean) {
        isDebugLoggingEnabled = enabled
        if (isDebugLoggingEnabled) {
            Log.d(TAG, "Debug logging enabled")
        }
    }

    // Private helper methods

    private fun generateModelId(): String {
        return MODEL_ID_PREFIX + UUID.randomUUID().toString().replace("-", "").substring(0, 8)
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

    private fun convertEValuesToTensors(eValues: Array<EValue>): List<TensorData?> {
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

}