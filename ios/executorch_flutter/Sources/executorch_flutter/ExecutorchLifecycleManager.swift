/**
 * ExecuTorch Lifecycle Manager - iOS Platform Resource Management
 *
 * This class manages the lifecycle of ExecuTorch resources on iOS, including
 * application state monitoring, memory pressure handling, and proper cleanup
 * during app transitions. It ensures optimal resource usage and prevents
 * memory leaks in the iOS environment.
 *
 * Features:
 * - Application lifecycle monitoring (background, foreground, termination)
 * - Memory pressure detection and response
 * - Automatic model disposal during memory warnings
 * - Resource cleanup coordination
 * - Performance monitoring and reporting
 */
import Foundation
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

/**
 * Manages ExecuTorch resource lifecycle in response to iOS app state changes
 */
class ExecutorchLifecycleManager {

    fileprivate static let TAG = "ExecutorchLifecycleManager"

    // Singleton instance
    static let shared = ExecutorchLifecycleManager()

    // Lifecycle state
    private var isActive = false
    #if os(iOS)
    private var backgroundTaskIdentifier: UIBackgroundTaskIdentifier = .invalid
    #endif

    // Model managers to coordinate with
    private var modelManagers: [WeakModelManagerRef] = []

    // Memory monitoring
    private var lastMemoryWarning: Date?
    private let memoryWarningCooldown: TimeInterval = 10.0 // seconds

    // Notification observers
    private var notificationObservers: [NSObjectProtocol] = []

    /**
     * Weak reference wrapper for model managers
     */
    private struct WeakModelManagerRef {
        weak var manager: ExecutorchModelManager?
    }

    private init() {
        setupApplicationLifecycleObservers()
        setupMemoryPressureObservers()
        print("[\(Self.TAG)] ExecutorchLifecycleManager initialized")
    }

    deinit {
        cleanup()
        print("[\(Self.TAG)] ExecutorchLifecycleManager deinitialized")
    }

    // MARK: - Public API

    /**
     * Register a model manager for lifecycle coordination
     */
    func registerModelManager(_ manager: ExecutorchModelManager) {
        // Clean up any deallocated references
        modelManagers.removeAll { $0.manager == nil }

        // Add new manager
        modelManagers.append(WeakModelManagerRef(manager: manager))
        print("[\(Self.TAG)] Registered model manager, total: \(modelManagers.count)")
    }

    /**
     * Unregister a model manager
     */
    func unregisterModelManager(_ manager: ExecutorchModelManager) {
        modelManagers.removeAll { $0.manager === manager }
        print("[\(Self.TAG)] Unregistered model manager, remaining: \(modelManagers.count)")
    }

    /**
     * Force cleanup of all resources
     */
    func forceCleanup() async {
        print("[\(Self.TAG)] Force cleanup requested")
        await disposeAllModels()
    }

    /**
     * Get current memory usage statistics
     */
    func getMemoryStats() -> MemoryStats {
        let usedMemory = getCurrentMemoryUsage()
        let availableMemory = ProcessInfo.processInfo.physicalMemory
        let memoryPressure = getMemoryPressureLevel()

        return MemoryStats(
            usedBytes: usedMemory,
            availableBytes: availableMemory,
            pressureLevel: memoryPressure,
            lastWarningTime: lastMemoryWarning
        )
    }

    // MARK: - Application Lifecycle Handling

    private func setupApplicationLifecycleObservers() {
        #if os(iOS)
        let notificationCenter = NotificationCenter.default

        // App will become active
        let willBecomeActiveObserver = notificationCenter.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillBecomeActive()
        }
        notificationObservers.append(willBecomeActiveObserver)

        // App did become active
        let didBecomeActiveObserver = notificationCenter.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppDidBecomeActive()
        }
        notificationObservers.append(didBecomeActiveObserver)

        // App will resign active
        let willResignActiveObserver = notificationCenter.addObserver(
            forName: UIApplication.willResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillResignActive()
        }
        notificationObservers.append(willResignActiveObserver)

        // App did enter background
        let didEnterBackgroundObserver = notificationCenter.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppDidEnterBackground()
        }
        notificationObservers.append(didEnterBackgroundObserver)

        // App will terminate
        let willTerminateObserver = notificationCenter.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillTerminate()
        }
        notificationObservers.append(willTerminateObserver)
        #elseif os(macOS)
        let notificationCenter = NotificationCenter.default

        // macOS: App will become active
        let didBecomeActiveObserver = notificationCenter.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppDidBecomeActive()
        }
        notificationObservers.append(didBecomeActiveObserver)

        // macOS: App will resign active
        let willResignActiveObserver = notificationCenter.addObserver(
            forName: NSApplication.willResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillResignActive()
        }
        notificationObservers.append(willResignActiveObserver)

        // macOS: App will terminate
        let willTerminateObserver = notificationCenter.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAppWillTerminate()
        }
        notificationObservers.append(willTerminateObserver)
        #endif
    }

    private func setupMemoryPressureObservers() {
        #if os(iOS)
        let notificationCenter = NotificationCenter.default

        // Memory warning
        let memoryWarningObserver = notificationCenter.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        notificationObservers.append(memoryWarningObserver)
        #elseif os(macOS)
        // macOS doesn't have the same memory warning API
        // Could monitor process memory manually if needed
        print("[\(Self.TAG)] Memory pressure monitoring not implemented for macOS")
        #endif
    }

    // MARK: - Lifecycle Event Handlers

    private func handleAppWillBecomeActive() {
        print("[\(Self.TAG)] App will become active")
        // Prepare for active use
    }

    private func handleAppDidBecomeActive() {
        print("[\(Self.TAG)] App did become active")
        isActive = true

        // End background task if running (iOS only)
        #if os(iOS)
        if backgroundTaskIdentifier != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskIdentifier)
            backgroundTaskIdentifier = .invalid
        }
        #endif
    }

    private func handleAppWillResignActive() {
        print("[\(Self.TAG)] App will resign active")
        isActive = false
    }

    private func handleAppDidEnterBackground() {
        print("[\(Self.TAG)] App did enter background")

        #if os(iOS)
        // Start background task to allow cleanup (iOS only)
        backgroundTaskIdentifier = UIApplication.shared.beginBackgroundTask(
            withName: "ExecutorchCleanup"
        ) { [weak self] in
            self?.handleBackgroundTaskExpiration()
        }
        #endif

        // Perform background cleanup
        Task {
            await performBackgroundCleanup()
        }
    }

    private func handleAppWillTerminate() {
        print("[\(Self.TAG)] App will terminate")
        Task {
            await disposeAllModels()
        }
    }

    private func handleBackgroundTaskExpiration() {
        print("[\(Self.TAG)] Background task expired")
        #if os(iOS)
        if backgroundTaskIdentifier != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskIdentifier)
            backgroundTaskIdentifier = .invalid
        }
        #endif
    }

    // MARK: - Memory Management

    private func handleMemoryWarning() {
        let now = Date()

        // Avoid too frequent memory warning handling
        if let lastWarning = lastMemoryWarning,
           now.timeIntervalSince(lastWarning) < memoryWarningCooldown {
            print("[\(Self.TAG)] Memory warning ignored (too frequent)")
            return
        }

        lastMemoryWarning = now
        print("[\(Self.TAG)] Memory warning received, performing cleanup")

        Task {
            await handleMemoryPressure()
        }
    }

    private func handleMemoryPressure() async {
        let memoryStats = getMemoryStats()
        print("[\(Self.TAG)] Memory pressure - used: \(memoryStats.usedBytes / 1024 / 1024)MB")

        // Dispose least recently used models
        let activeManagers = modelManagers.compactMap { $0.manager }
        for manager in activeManagers {
            // Let each manager handle its own memory pressure response
            await manager.handleMemoryPressure()
        }

        // Log memory usage after cleanup
        let newMemoryStats = getMemoryStats()
        print("[\(Self.TAG)] After cleanup - used: \(newMemoryStats.usedBytes / 1024 / 1024)MB")
    }

    private func performBackgroundCleanup() async {
        print("[\(Self.TAG)] Performing background cleanup")

        // Dispose non-essential models to free memory
        let activeManagers = modelManagers.compactMap { $0.manager }
        for manager in activeManagers {
            await manager.performBackgroundCleanup()
        }

        // End background task (iOS only)
        #if os(iOS)
        if backgroundTaskIdentifier != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskIdentifier)
            backgroundTaskIdentifier = .invalid
        }
        #endif
    }

    private func disposeAllModels() async {
        print("[\(Self.TAG)] Disposing all models")

        let activeManagers = modelManagers.compactMap { $0.manager }
        for manager in activeManagers {
            await manager.disposeAllModels()
        }

        modelManagers.removeAll()
    }

    // MARK: - Memory Monitoring Utilities

    private func getCurrentMemoryUsage() -> UInt64 {
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

        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    private func getMemoryPressureLevel() -> MemoryPressureLevel {
        let usedMemory = getCurrentMemoryUsage()
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usageRatio = Double(usedMemory) / Double(totalMemory)

        switch usageRatio {
        case 0..<0.6:
            return .low
        case 0.6..<0.8:
            return .medium
        case 0.8..<0.95:
            return .high
        default:
            return .critical
        }
    }

    private func cleanup() {
        // Remove notification observers
        for observer in notificationObservers {
            NotificationCenter.default.removeObserver(observer)
        }
        notificationObservers.removeAll()

        // End background task if active (iOS only)
        #if os(iOS)
        if backgroundTaskIdentifier != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTaskIdentifier)
            backgroundTaskIdentifier = .invalid
        }
        #endif
    }
}

// MARK: - Extension for Model Manager Integration

extension ExecutorchModelManager {

    /**
     * Handle memory pressure by disposing least recently used models
     */
    func handleMemoryPressure() async {
        print("[\(Self.TAG)] Handling memory pressure")

        // In a real implementation, this would dispose LRU models
        // For now, we'll dispose half of the loaded models
        let loadedIds = try? getLoadedModelIds()
        let modelsToDispose = loadedIds?.prefix(loadedIds!.count / 2) ?? []

        for modelId in modelsToDispose {
            do {
                try await disposeModel(modelId: modelId)
                print("[\(Self.TAG)] Disposed model due to memory pressure: \(modelId)")
            } catch {
                print("[\(Self.TAG)] Failed to dispose model during memory pressure: \(modelId)")
            }
        }
    }

    /**
     * Perform background cleanup tasks
     */
    func performBackgroundCleanup() async {
        print("[\(Self.TAG)] Performing background cleanup")

        // In background, dispose non-essential models to free resources
        // This is a conservative approach to maintain app responsiveness
        await handleMemoryPressure()
    }
}

// MARK: - Supporting Data Types

/**
 * Memory usage statistics
 */
struct MemoryStats {
    let usedBytes: UInt64
    let availableBytes: UInt64
    let pressureLevel: MemoryPressureLevel
    let lastWarningTime: Date?

    var usageMB: Double {
        return Double(usedBytes) / (1024 * 1024)
    }

    var availableMB: Double {
        return Double(availableBytes) / (1024 * 1024)
    }

    var usagePercentage: Double {
        return availableBytes > 0 ? Double(usedBytes) / Double(availableBytes) * 100 : 0
    }
}

/**
 * Memory pressure levels
 */
enum MemoryPressureLevel {
    case low
    case medium
    case high
    case critical

    var description: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .critical: return "Critical"
        }
    }
}