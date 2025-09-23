/**
 * ExecuTorch Lifecycle Manager - Android Platform Resource Management
 *
 * This class manages the lifecycle of ExecuTorch resources on Android, including
 * application state monitoring, memory pressure handling, and proper cleanup
 * during app transitions. It ensures optimal resource usage and prevents
 * memory leaks in the Android environment.
 *
 * Features:
 * - Application lifecycle monitoring (background, foreground, low memory)
 * - Memory pressure detection and response
 * - Automatic model disposal during memory warnings
 * - Resource cleanup coordination
 * - Performance monitoring and reporting
 */
package com.zcreations.executorch_flutter

import android.app.Activity
import android.app.Application
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import kotlinx.coroutines.*
import java.lang.ref.WeakReference
import java.util.concurrent.ConcurrentHashMap

// Import Pigeon generated API
import com.zcreations.executorch_flutter.generated.*

/**
 * Manages ExecuTorch resource lifecycle in response to Android app state changes
 */
class ExecutorchLifecycleManager private constructor(
    private val application: Application
) : Application.ActivityLifecycleCallbacks, ComponentCallbacks2 {

    companion object {
        private const val TAG = "ExecutorchLifecycleManager"
        private const val MEMORY_WARNING_COOLDOWN_MS = 10_000L // 10 seconds

        @Volatile
        private var INSTANCE: ExecutorchLifecycleManager? = null

        fun getInstance(application: Application): ExecutorchLifecycleManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ExecutorchLifecycleManager(application).also {
                    INSTANCE = it
                    application.registerActivityLifecycleCallbacks(it)
                    application.registerComponentCallbacks(it)
                }
            }
        }
    }

    // Lifecycle state
    private var isInForeground = false
    private var activeActivityCount = 0
    private var lastMemoryWarning = 0L

    // Model managers to coordinate with
    private val modelManagers = ConcurrentHashMap<String, WeakReference<ExecutorchModelManager>>()

    // Coroutine scope for lifecycle operations
    private val lifecycleScope = CoroutineScope(
        Dispatchers.Default + SupervisorJob() + CoroutineName("ExecutorchLifecycle")
    )

    init {
        Log.d(TAG, "ExecutorchLifecycleManager initialized")
    }

    // MARK: - Public API

    /**
     * Register a model manager for lifecycle coordination
     */
    fun registerModelManager(id: String, manager: ExecutorchModelManager) {
        // Clean up any deallocated references
        cleanupDeadReferences()

        // Add new manager
        modelManagers[id] = WeakReference(manager)
        Log.d(TAG, "Registered model manager: $id, total: ${modelManagers.size}")
    }

    /**
     * Unregister a model manager
     */
    fun unregisterModelManager(id: String) {
        modelManagers.remove(id)
        Log.d(TAG, "Unregistered model manager: $id, remaining: ${modelManagers.size}")
    }

    /**
     * Force cleanup of all resources
     */
    suspend fun forceCleanup() {
        Log.d(TAG, "Force cleanup requested")
        disposeAllModels()
    }

    /**
     * Get current memory usage statistics
     */
    fun getMemoryStats(): AndroidMemoryStats {
        val runtime = Runtime.getRuntime()
        val activityManager = application.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        return AndroidMemoryStats(
            usedBytes = runtime.totalMemory() - runtime.freeMemory(),
            maxBytes = runtime.maxMemory(),
            availableBytes = memInfo.availMem,
            totalSystemBytes = memInfo.totalMem,
            isLowMemory = memInfo.lowMemory,
            lastWarningTime = lastMemoryWarning
        )
    }

    /**
     * Check if app is currently in foreground
     */
    fun isAppInForeground(): Boolean = isInForeground

    // MARK: - Activity Lifecycle Callbacks

    override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
        Log.d(TAG, "Activity created: ${activity.localClassName}")
    }

    override fun onActivityStarted(activity: Activity) {
        activeActivityCount++
        updateForegroundState()
        Log.d(TAG, "Activity started: ${activity.localClassName}, active count: $activeActivityCount")
    }

    override fun onActivityResumed(activity: Activity) {
        Log.d(TAG, "Activity resumed: ${activity.localClassName}")
    }

    override fun onActivityPaused(activity: Activity) {
        Log.d(TAG, "Activity paused: ${activity.localClassName}")
    }

    override fun onActivityStopped(activity: Activity) {
        activeActivityCount--
        updateForegroundState()
        Log.d(TAG, "Activity stopped: ${activity.localClassName}, active count: $activeActivityCount")

        if (activeActivityCount == 0) {
            handleAppEnteredBackground()
        }
    }

    override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {
        Log.d(TAG, "Activity saving state: ${activity.localClassName}")
    }

    override fun onActivityDestroyed(activity: Activity) {
        Log.d(TAG, "Activity destroyed: ${activity.localClassName}")
    }

    // MARK: - Component Callbacks

    override fun onConfigurationChanged(newConfig: Configuration) {
        Log.d(TAG, "Configuration changed")
    }

    override fun onLowMemory() {
        Log.w(TAG, "System low memory warning")
        handleMemoryPressure(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
    }

    override fun onTrimMemory(level: Int) {
        Log.w(TAG, "Memory trim requested, level: $level")
        handleMemoryPressure(level)
    }

    // MARK: - Private Implementation

    private fun updateForegroundState() {
        val wasInForeground = isInForeground
        isInForeground = activeActivityCount > 0

        if (wasInForeground != isInForeground) {
            if (isInForeground) {
                handleAppEnteredForeground()
            }
        }
    }

    private fun handleAppEnteredForeground() {
        Log.d(TAG, "App entered foreground")
        // Restore any paused operations if needed
    }

    private fun handleAppEnteredBackground() {
        Log.d(TAG, "App entered background")

        // TODO: Consider less aggressive background cleanup for demo apps
        // For now, skip automatic model disposal to prevent user frustration
        // lifecycleScope.launch {
        //     performBackgroundCleanup()
        // }
    }

    private fun handleMemoryPressure(level: Int) {
        val now = System.currentTimeMillis()

        // Avoid too frequent memory warning handling
        if (now - lastMemoryWarning < MEMORY_WARNING_COOLDOWN_MS) {
            Log.d(TAG, "Memory pressure ignored (too frequent)")
            return
        }

        lastMemoryWarning = now
        Log.w(TAG, "Memory pressure level: $level")

        lifecycleScope.launch {
            when {
                level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> {
                    // Critical memory pressure - dispose most models
                    handleCriticalMemoryPressure()
                }
                level >= ComponentCallbacks2.TRIM_MEMORY_MODERATE -> {
                    // Moderate memory pressure - dispose some models
                    handleModerateMemoryPressure()
                }
                level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> {
                    // App in background - conservative cleanup
                    handleBackgroundMemoryPressure()
                }
                else -> {
                    // Light memory pressure - minimal cleanup
                    handleLightMemoryPressure()
                }
            }
        }
    }

    private suspend fun handleCriticalMemoryPressure() {
        Log.w(TAG, "Handling critical memory pressure")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.handleMemoryPressure(severity = MemoryPressureSeverity.CRITICAL)
        }

        // Force garbage collection
        System.gc()
    }

    private suspend fun handleModerateMemoryPressure() {
        Log.w(TAG, "Handling moderate memory pressure")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.handleMemoryPressure(severity = MemoryPressureSeverity.MODERATE)
        }
    }

    private suspend fun handleBackgroundMemoryPressure() {
        Log.d(TAG, "Handling background memory pressure")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.handleMemoryPressure(severity = MemoryPressureSeverity.BACKGROUND)
        }
    }

    private suspend fun handleLightMemoryPressure() {
        Log.d(TAG, "Handling light memory pressure")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.handleMemoryPressure(severity = MemoryPressureSeverity.LIGHT)
        }
    }

    private suspend fun performBackgroundCleanup() {
        Log.d(TAG, "Performing background cleanup")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.performBackgroundCleanup()
        }
    }

    private suspend fun disposeAllModels() {
        Log.d(TAG, "Disposing all models")

        val activeManagers = getActiveManagers()
        for (manager in activeManagers) {
            manager.disposeAllModels()
        }

        modelManagers.clear()
    }

    private fun getActiveManagers(): List<ExecutorchModelManager> {
        cleanupDeadReferences()
        return modelManagers.values.mapNotNull { it.get() }
    }

    private fun cleanupDeadReferences() {
        val deadKeys = modelManagers.entries
            .filter { it.value.get() == null }
            .map { it.key }

        deadKeys.forEach { modelManagers.remove(it) }

        if (deadKeys.isNotEmpty()) {
            Log.d(TAG, "Cleaned up ${deadKeys.size} dead model manager references")
        }
    }

    fun cleanup() {
        lifecycleScope.cancel()
        application.unregisterActivityLifecycleCallbacks(this)
        application.unregisterComponentCallbacks(this)
        Log.d(TAG, "ExecutorchLifecycleManager cleaned up")
    }
}

// MARK: - Extension for Model Manager Integration

/**
 * Memory pressure severity levels
 */
enum class MemoryPressureSeverity {
    LIGHT,      // Minimal cleanup needed
    BACKGROUND, // App in background, moderate cleanup
    MODERATE,   // Dispose some models
    CRITICAL    // Dispose most models immediately
}

/**
 * Extension functions for ExecutorchModelManager to handle lifecycle events
 */
suspend fun ExecutorchModelManager.handleMemoryPressure(severity: MemoryPressureSeverity) {
    Log.d("ExecutorchModelManager", "Handling memory pressure: $severity")

    when (severity) {
        MemoryPressureSeverity.CRITICAL -> {
            // Dispose 75% of models
            val loadedIds = getLoadedModelIds()
            val modelsToDispose = loadedIds.take(loadedIds.size * 3 / 4)
            disposeModels(modelsToDispose)
        }
        MemoryPressureSeverity.MODERATE -> {
            // Dispose 50% of models
            val loadedIds = getLoadedModelIds()
            val modelsToDispose = loadedIds.take(loadedIds.size / 2)
            disposeModels(modelsToDispose)
        }
        MemoryPressureSeverity.BACKGROUND -> {
            // Dispose 25% of models
            val loadedIds = getLoadedModelIds()
            val modelsToDispose = loadedIds.take(loadedIds.size / 4)
            disposeModels(modelsToDispose)
        }
        MemoryPressureSeverity.LIGHT -> {
            // Just request GC
            TensorMemoryManager.requestGCIfNeeded(threshold = 0.7)
        }
    }
}

suspend fun ExecutorchModelManager.performBackgroundCleanup() {
    Log.d("ExecutorchModelManager", "Performing background cleanup")
    handleMemoryPressure(MemoryPressureSeverity.BACKGROUND)
}

private suspend fun ExecutorchModelManager.disposeModels(modelIds: List<String>) {
    for (modelId in modelIds) {
        try {
            disposeModel(modelId)
            Log.d("ExecutorchModelManager", "Disposed model due to memory pressure: $modelId")
        } catch (e: Exception) {
            Log.e("ExecutorchModelManager", "Failed to dispose model during memory pressure: $modelId", e)
        }
    }
}

// MARK: - Supporting Data Types

/**
 * Android-specific memory usage statistics
 */
data class AndroidMemoryStats(
    val usedBytes: Long,
    val maxBytes: Long,
    val availableBytes: Long,
    val totalSystemBytes: Long,
    val isLowMemory: Boolean,
    val lastWarningTime: Long
) {
    val usageMB: Double
        get() = usedBytes.toDouble() / (1024 * 1024)

    val maxMB: Double
        get() = maxBytes.toDouble() / (1024 * 1024)

    val availableMB: Double
        get() = availableBytes.toDouble() / (1024 * 1024)

    val usagePercentage: Double
        get() = if (maxBytes > 0) usedBytes.toDouble() / maxBytes.toDouble() * 100 else 0.0

    val systemUsagePercentage: Double
        get() = if (totalSystemBytes > 0) (totalSystemBytes - availableBytes).toDouble() / totalSystemBytes.toDouble() * 100 else 0.0
}