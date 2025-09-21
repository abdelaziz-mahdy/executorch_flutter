/**
 * ExecuTorch Tensor Utilities - Android Platform Extensions
 *
 * This file provides utility functions and extensions for tensor data validation,
 * conversion, and manipulation specific to the Android ExecuTorch integration.
 * It includes shape validation, data type checks, and memory-efficient operations.
 *
 * Features:
 * - Tensor shape and data type validation
 * - Memory-efficient tensor operations
 * - ExecuTorch tensor format validation
 * - Performance monitoring utilities
 * - Error handling with detailed diagnostics
 */
package com.zcreations.executorch_flutter

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

// Import Pigeon generated API
import com.zcreations.executorch_flutter.generated.*

/**
 * Utility extensions for TensorData validation and operations
 */
fun TensorData.validate() {
    // Validate shape consistency
    require(shape.isNotEmpty()) { "Tensor shape cannot be empty" }
    require(shape.all { it != null && it > 0 }) { "All tensor dimensions must be positive" }

    // Calculate expected data size
    val expectedElements = shape.filterNotNull().fold(1L) { acc, dim -> acc * dim }
    val bytesPerElement = dataType.bytesPerElement()
    val expectedBytes = expectedElements.toInt() * bytesPerElement

    require(data.size == expectedBytes) {
        "Tensor data size mismatch: expected $expectedBytes bytes, got ${data.size} bytes"
    }

    // Validate data type constraints
    dataType.validateData(data)
}

/**
 * Get total number of elements in the tensor
 */
val TensorData.elementCount: Long
    get() = shape.filterNotNull().fold(1L) { acc, dim -> acc * dim }

/**
 * Get total size in bytes
 */
val TensorData.sizeInBytes: Int
    get() = (elementCount * dataType.bytesPerElement()).toInt()

/**
 * Check if tensor is scalar (single element)
 */
val TensorData.isScalar: Boolean
    get() = elementCount == 1L

/**
 * Check if tensor is vector (1D)
 */
val TensorData.isVector: Boolean
    get() = shape.size == 1

/**
 * Check if tensor is matrix (2D)
 */
val TensorData.isMatrix: Boolean
    get() = shape.size == 2

/**
 * Utility extensions for TensorType operations
 */
fun TensorType.bytesPerElement(): Int {
    return when (this) {
        TensorType.FLOAT32 -> 4
        TensorType.INT32 -> 4
        TensorType.INT8 -> 1
        TensorType.UINT8 -> 1
    }
}

/**
 * Validate data consistency for this tensor type
 */
fun TensorType.validateData(data: ByteArray) {
    val expectedSize = data.size
    val elementSize = bytesPerElement()

    require(expectedSize % elementSize == 0) {
        "Data size $expectedSize is not aligned to element size $elementSize for type $this"
    }

    // Additional type-specific validations
    when (this) {
        TensorType.FLOAT32 -> validateFloat32Data(data)
        TensorType.INT32 -> validateInt32Data(data)
        TensorType.INT8, TensorType.UINT8 -> {
            // Byte data is always valid
        }
    }
}

private fun validateFloat32Data(data: ByteArray) {
    val buffer = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asFloatBuffer()
    val floats = FloatArray(buffer.remaining())
    buffer.get(floats)

    for (i in floats.indices) {
        val value = floats[i]
        if (value.isNaN() || value.isInfinite()) {
            // Note: This is a warning, not an error, as some models may use NaN/Inf intentionally
            Log.w("ExecutorchTensorUtils", "Warning: Found NaN/Inf value at index $i")
        }
    }
}

private fun validateInt32Data(data: ByteArray) {
    // Int32 validation is typically just alignment check
    // Additional domain-specific validations could be added here
}

/**
 * Shape validation utilities
 */
object TensorShapeValidator {

    /**
     * Validate that two tensors have compatible shapes for element-wise operations
     */
    fun validateCompatibleShapes(shape1: List<Int>, shape2: List<Int>) {
        require(shape1.size == shape2.size) {
            "Shape dimension mismatch: ${shape1.size} vs ${shape2.size}"
        }

        for (i in shape1.indices) {
            require(shape1[i] == shape2[i]) {
                "Shape mismatch at dimension $i: ${shape1[i]} vs ${shape2[i]}"
            }
        }
    }

    /**
     * Validate that a shape can be reshaped to another shape
     */
    fun validateReshapeCompatibility(oldShape: List<Int>, newShape: List<Int>) {
        val oldElements = oldShape.fold(1) { acc, dim -> acc * dim }
        val newElements = newShape.fold(1) { acc, dim -> acc * dim }

        require(oldElements == newElements) {
            "Cannot reshape tensor with $oldElements elements to shape with $newElements elements"
        }
    }

    /**
     * Validate common tensor shapes for mobile ML models
     */
    fun validateMobileModelShape(shape: List<Int>) {
        // Common validations for mobile models
        require(shape.size in 1..5) {
            "Mobile models typically use 1-5 dimensional tensors, got ${shape.size} dimensions"
        }

        // Check for reasonable tensor sizes (mobile memory constraints)
        val elements = shape.fold(1) { acc, dim -> acc * dim }
        val maxElements = 100_000_000 // ~400MB for float32

        require(elements <= maxElements) {
            "Tensor too large for mobile device: $elements elements (max: $maxElements)"
        }

        // Validate batch dimension (typically first dimension)
        if (shape.size > 1 && shape[0] > 64) {
            Log.w("TensorShapeValidator", "Warning: Large batch size ${shape[0]} may cause memory issues on mobile")
        }
    }
}

/**
 * Performance monitoring utilities for tensor operations
 */
object TensorPerformanceMonitor {

    /**
     * Measure tensor operation execution time
     */
    inline fun <T> measureOperation(operation: () -> T): Pair<T, Double> {
        val startTime = System.nanoTime()
        val result = operation()
        val endTime = System.nanoTime()
        val timeMs = (endTime - startTime) / 1_000_000.0

        return Pair(result, timeMs)
    }

    /**
     * Log tensor memory usage and performance metrics
     */
    fun logTensorMetrics(tensor: TensorData, operation: String) {
        val sizeKB = tensor.sizeInBytes.toDouble() / 1024.0
        val sizeMB = sizeKB / 1024.0

        Log.d("TensorPerformance", "$operation: shape=${tensor.shape}, " +
                "type=${tensor.dataType}, size=${"%.2f".format(sizeMB)}MB")
    }

    /**
     * Estimate memory requirements for a tensor operation
     */
    fun estimateMemoryUsage(inputShapes: List<List<Int>>, outputShapes: List<List<Int>>, dataType: TensorType): Int {
        val inputElements = inputShapes.sumOf { it.fold(1L) { acc, dim -> acc * dim } }
        val outputElements = outputShapes.sumOf { it.fold(1L) { acc, dim -> acc * dim } }
        val totalElements = inputElements + outputElements

        return (totalElements * dataType.bytesPerElement()).toInt()
    }
}

/**
 * Tensor data conversion utilities
 */
object TensorDataConverter {

    /**
     * Convert ByteArray to typed array with proper byte order handling
     */
    fun byteArrayToFloatArray(data: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val result = FloatArray(buffer.remaining())
        buffer.get(result)
        return result
    }

    fun byteArrayToIntArray(data: ByteArray): IntArray {
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asIntBuffer()
        val result = IntArray(buffer.remaining())
        buffer.get(result)
        return result
    }

    /**
     * Convert typed array to ByteArray with proper byte order
     */
    fun floatArrayToByteArray(array: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(array.size * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(array)
        return buffer.array()
    }

    fun intArrayToByteArray(array: IntArray): ByteArray {
        val buffer = ByteBuffer.allocate(array.size * 4).order(ByteOrder.nativeOrder())
        buffer.asIntBuffer().put(array)
        return buffer.array()
    }

    /**
     * Safe data conversion with validation
     */
    fun convertDataSafely(data: ByteArray, elementSize: Int): ByteArray {
        require(data.size % elementSize == 0) {
            "Data size ${data.size} not aligned to element size $elementSize"
        }
        return data
    }
}

/**
 * Memory management utilities for tensor operations
 */
object TensorMemoryManager {

    private const val TAG = "TensorMemoryManager"

    /**
     * Check available memory before tensor operations
     */
    fun checkAvailableMemory(requiredBytes: Int) {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        val freeMemory = maxMemory - usedMemory

        // Require at least 2x the required memory to be safe
        val safeRequiredMemory = requiredBytes * 2

        require(freeMemory > safeRequiredMemory) {
            "Insufficient memory: need $safeRequiredMemory, available $freeMemory"
        }
    }

    /**
     * Get current memory usage statistics
     */
    fun getMemoryStats(): MemoryStats {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val usedMemory = totalMemory - freeMemory

        return MemoryStats(
            usedBytes = usedMemory,
            maxBytes = maxMemory,
            freeBytes = freeMemory,
            totalBytes = totalMemory
        )
    }

    /**
     * Log current memory usage for debugging
     */
    fun logMemoryUsage(context: String = "") {
        val stats = getMemoryStats()
        val usedMB = stats.usedBytes.toDouble() / (1024 * 1024)
        val maxMB = stats.maxBytes.toDouble() / (1024 * 1024)
        Log.d(TAG, "$context: Used memory: ${"%.2f".format(usedMB)}MB / ${"%.2f".format(maxMB)}MB")
    }

    /**
     * Request garbage collection if memory usage is high
     */
    fun requestGCIfNeeded(threshold: Double = 0.8) {
        val stats = getMemoryStats()
        val usageRatio = stats.usedBytes.toDouble() / stats.maxBytes.toDouble()

        if (usageRatio > threshold) {
            Log.d(TAG, "Memory usage high (${"%.1f".format(usageRatio * 100)}%), requesting GC")
            System.gc()
        }
    }
}

/**
 * Memory usage statistics data class
 */
data class MemoryStats(
    val usedBytes: Long,
    val maxBytes: Long,
    val freeBytes: Long,
    val totalBytes: Long
) {
    val usageMB: Double
        get() = usedBytes.toDouble() / (1024 * 1024)

    val maxMB: Double
        get() = maxBytes.toDouble() / (1024 * 1024)

    val usagePercentage: Double
        get() = if (maxBytes > 0) usedBytes.toDouble() / maxBytes.toDouble() * 100 else 0.0
}