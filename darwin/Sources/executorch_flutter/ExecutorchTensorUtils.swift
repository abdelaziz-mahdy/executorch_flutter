/**
 * ExecuTorch Tensor Utilities - iOS Platform Extensions
 *
 * This file provides utility functions and extensions for tensor data validation,
 * conversion, and manipulation specific to the iOS ExecuTorch integration.
 * It includes shape validation, data type checks, and memory-efficient operations.
 *
 * Features:
 * - Tensor shape and data type validation
 * - Memory-efficient tensor operations
 * - ExecuTorch tensor format validation
 * - Performance monitoring utilities
 * - Error handling with detailed diagnostics
 */
import Foundation

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
 * Utility extensions for TensorData validation and operations
 */
extension TensorData {

    /**
     * Validate tensor data consistency and constraints
     */
    func validate() throws {
        // Validate shape consistency
        guard !shape.isEmpty else {
            throw ExecutorchError.validationError("Tensor shape cannot be empty")
        }

        // Unwrap optionals and validate
        let unwrappedShape = shape.compactMap { $0 }
        guard unwrappedShape.count == shape.count else {
            throw ExecutorchError.validationError("Tensor shape contains nil values")
        }

        guard unwrappedShape.allSatisfy({ $0 > 0 }) else {
            throw ExecutorchError.validationError("All tensor dimensions must be positive")
        }

        // Calculate expected data size
        let expectedElements = unwrappedShape.reduce(1, *)
        let bytesPerElement = dataType.bytesPerElement()
        let expectedBytes = Int(expectedElements) * bytesPerElement

        guard data.data.count == expectedBytes else {
            throw ExecutorchError.validationError(
                "Tensor data size mismatch: expected \(expectedBytes) bytes, got \(data.data.count) bytes"
            )
        }

        // Validate data type constraints
        try dataType.validateData(data.data)
    }

    /**
     * Get total number of elements in the tensor
     */
    var elementCount: Int64 {
        return shape.compactMap { $0 }.reduce(1, *)
    }

    /**
     * Get total size in bytes
     */
    var sizeInBytes: Int {
        return Int(elementCount) * dataType.bytesPerElement()
    }

    /**
     * Check if tensor is scalar (single element)
     */
    var isScalar: Bool {
        return elementCount == 1
    }

    /**
     * Check if tensor is vector (1D)
     */
    var isVector: Bool {
        return shape.count == 1
    }

    /**
     * Check if tensor is matrix (2D)
     */
    var isMatrix: Bool {
        return shape.count == 2
    }
}

/**
 * Utility extensions for TensorType operations
 */
extension TensorType {

    /**
     * Get the number of bytes per element for this data type
     */
    func bytesPerElement() -> Int {
        switch self {
        case .float32:
            return 4
        case .int32:
            return 4
        case .int8:
            return 1
        case .uint8:
            return 1
        }
    }

    /**
     * Validate data consistency for this tensor type
     */
    func validateData(_ data: Data) throws {
        let expectedSize = data.count
        let elementSize = bytesPerElement()

        guard expectedSize % elementSize == 0 else {
            throw ExecutorchError.validationError(
                "Data size \(expectedSize) is not aligned to element size \(elementSize) for type \(self)"
            )
        }

        // Additional type-specific validations
        switch self {
        case .float32:
            try validateFloat32Data(data)
        case .int32:
            try validateInt32Data(data)
        case .int8, .uint8:
            // Byte data is always valid
            break
        }
    }

    private func validateFloat32Data(_ data: Data) throws {
        let floatCount = data.count / 4
        data.withUnsafeBytes { bytes in
            let floats = bytes.bindMemory(to: Float.self)
            for i in 0..<floatCount {
                let value = floats[i]
                if value.isNaN || value.isInfinite {
                    // Note: This is a warning, not an error, as some models may use NaN/Inf intentionally
                    print("[ExecutorchTensorUtils] Warning: Found NaN/Inf value at index \(i)")
                }
            }
        }
    }

    private func validateInt32Data(_ data: Data) throws {
        // Int32 validation is typically just alignment check
        // Additional domain-specific validations could be added here
    }
}

/**
 * Shape validation utilities
 */
struct TensorShapeValidator {

    /**
     * Validate that two tensors have compatible shapes for element-wise operations
     */
    static func validateCompatibleShapes(_ shape1: [Int], _ shape2: [Int]) throws {
        guard shape1.count == shape2.count else {
            throw ExecutorchError.validationError(
                "Shape dimension mismatch: \(shape1.count) vs \(shape2.count)"
            )
        }

        for (i, (dim1, dim2)) in zip(shape1, shape2).enumerated() {
            guard dim1 == dim2 else {
                throw ExecutorchError.validationError(
                    "Shape mismatch at dimension \(i): \(dim1) vs \(dim2)"
                )
            }
        }
    }

    /**
     * Validate that a shape can be reshaped to another shape
     */
    static func validateReshapeCompatibility(from oldShape: [Int], to newShape: [Int]) throws {
        let oldElements = oldShape.reduce(1, *)
        let newElements = newShape.reduce(1, *)

        guard oldElements == newElements else {
            throw ExecutorchError.validationError(
                "Cannot reshape tensor with \(oldElements) elements to shape with \(newElements) elements"
            )
        }
    }

    /**
     * Validate common tensor shapes for mobile ML models
     */
    static func validateMobileModelShape(_ shape: [Int]) throws {
        // Common validations for mobile models
        guard shape.count >= 1 && shape.count <= 5 else {
            throw ExecutorchError.validationError(
                "Mobile models typically use 1-5 dimensional tensors, got \(shape.count) dimensions"
            )
        }

        // Check for reasonable tensor sizes (mobile memory constraints)
        let elements = shape.reduce(1, *)
        let maxElements = 100_000_000 // ~400MB for float32

        guard elements <= maxElements else {
            throw ExecutorchError.validationError(
                "Tensor too large for mobile device: \(elements) elements (max: \(maxElements))"
            )
        }

        // Validate batch dimension (typically first dimension)
        if shape.count > 1 && shape[0] > 64 {
            print("[TensorShapeValidator] Warning: Large batch size \(shape[0]) may cause memory issues on mobile")
        }
    }
}

/**
 * Performance monitoring utilities for tensor operations
 */
struct TensorPerformanceMonitor {

    /**
     * Measure tensor operation execution time
     */
    static func measureOperation<T>(_ operation: () throws -> T) rethrows -> (result: T, timeMs: Double) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try operation()
        let endTime = CFAbsoluteTimeGetCurrent()
        let timeMs = (endTime - startTime) * 1000

        return (result: result, timeMs: timeMs)
    }

    /**
     * Log tensor memory usage and performance metrics
     */
    static func logTensorMetrics(_ tensor: TensorData, operation: String) {
        let sizeKB = Double(tensor.sizeInBytes) / 1024.0
        let sizeMB = sizeKB / 1024.0

        print("[TensorPerformance] \(operation): shape=\(tensor.shape), " +
              "type=\(tensor.dataType), size=\(String(format: "%.2f", sizeMB))MB")
    }

    /**
     * Estimate memory requirements for a tensor operation
     */
    static func estimateMemoryUsage(inputShapes: [[Int]], outputShapes: [[Int]], dataType: TensorType) -> Int {
        let inputElements = inputShapes.map { $0.reduce(1, *) }.reduce(0, +)
        let outputElements = outputShapes.map { $0.reduce(1, *) }.reduce(0, +)
        let totalElements = inputElements + outputElements

        return totalElements * dataType.bytesPerElement()
    }
}

/**
 * Tensor data conversion utilities
 */
struct TensorDataConverter {

    /**
     * Convert Data to typed array with proper byte order handling
     */
    static func dataToFloatArray(_ data: Data) -> [Float] {
        return data.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: Float.self)
            return Array(buffer)
        }
    }

    static func dataToInt32Array(_ data: Data) -> [Int32] {
        return data.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: Int32.self)
            return Array(buffer)
        }
    }

    /**
     * Convert typed array to Data with proper byte order
     */
    static func floatArrayToData(_ array: [Float]) -> Data {
        return array.withUnsafeBytes { Data($0) }
    }

    static func int32ArrayToData(_ array: [Int32]) -> Data {
        return array.withUnsafeBytes { Data($0) }
    }

    /**
     * Safe data conversion with validation
     */
    static func convertDataSafely<T>(_ data: Data, to type: T.Type) throws -> [T] where T: FixedWidthInteger {
        let elementSize = MemoryLayout<T>.size
        guard data.count % elementSize == 0 else {
            throw ExecutorchError.validationError(
                "Data size \(data.count) not aligned to element size \(elementSize)"
            )
        }

        return data.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: T.self)
            return Array(buffer)
        }
    }
}

/**
 * Memory management utilities for tensor operations
 */
struct TensorMemoryManager {

    /**
     * Check available memory before tensor operations
     */
    static func checkAvailableMemory(requiredBytes: Int) throws {
        let availableMemory = ProcessInfo.processInfo.physicalMemory
        let usedMemory = getUsedMemory()
        let freeMemory = availableMemory - usedMemory

        // Require at least 2x the required memory to be safe
        let safeRequiredMemory = requiredBytes * 2

        guard freeMemory > safeRequiredMemory else {
            throw ExecutorchError.memoryError(
                "Insufficient memory: need \(safeRequiredMemory), available \(freeMemory)"
            )
        }
    }

    private static func getUsedMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return info.resident_size
        } else {
            return 0
        }
    }

    /**
     * Log current memory usage for debugging
     */
    static func logMemoryUsage(context: String = "") {
        let usedMemory = getUsedMemory()
        let usedMB = Double(usedMemory) / (1024 * 1024)
        print("[TensorMemoryManager] \(context): Used memory: \(String(format: "%.2f", usedMB))MB")
    }
}