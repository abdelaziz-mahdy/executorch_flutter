/**
 * ExecuTorch Wasm JavaScript Wrapper
 *
 * Provides a clean JavaScript API around the Emscripten-generated Module
 * for loading models, running inference, and managing resources.
 *
 * This wrapper is designed to be called from Dart via dart:js_interop.
 */

class ExecuTorchRunner {
  constructor() {
    this.module = null;
    this.isInitialized = false;
    this.nextModelId = 0;
    this.loadedModels = new Map(); // modelId -> { path, metadata }
    this.initPromise = null;
    this.debugLoggingEnabled = false; // Debug logging flag
  }

  /**
   * Enable or disable debug logging
   * @param {boolean} enabled - true to enable, false to disable
   */
  setDebugLogging(enabled) {
    this.debugLoggingEnabled = enabled;
    if (enabled) {
      console.log('[ExecuTorch] Debug logging enabled');
    } else {
      console.log('[ExecuTorch] Debug logging disabled');
    }
  }

  /**
   * Internal logging method (respects debug flag)
   * @param {...any} args - Arguments to log
   */
  _log(...args) {
    if (this.debugLoggingEnabled) {
      console.log('[ExecuTorch]', ...args);
    }
  }

  /**
   * Internal error logging method (always logs)
   * @param {...any} args - Arguments to log
   */
  _logError(...args) {
    console.error('[ExecuTorch Error]', ...args);
  }

  /**
   * Initialize the Wasm module (call once before using)
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.isInitialized) {
      return;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      // Configure Module before loading executor_runner.js
      window.Module = {
        // Redirect stdout/stderr to console
        print: (text) => {
          this._log(text);
        },
        printErr: (text) => {
          this._logError(text);
        },

        // Called when Wasm is ready
        onRuntimeInitialized: () => {
          this.module = window.Module;
          this.isInitialized = true;
          this._log('Wasm runtime initialized');
          resolve();
        },

        // Error handler
        onAbort: (error) => {
          this._logError('Wasm initialization failed:', error);
          reject(new Error(`Wasm initialization failed: ${error}`));
        },

        // Disable default UI elements
        canvas: null,
        setStatus: () => {},
        monitorRunDependencies: () => {},
      };

      // Load executor_runner.js (which loads the Wasm)
      // Path is relative to app's web directory
      const script = document.createElement('script');
      script.src = 'wasm/executor_runner.js';
      script.async = true;
      script.onload = () => {
        this._log('executor_runner.js loaded');
        // Module.onRuntimeInitialized will be called when ready
      };
      script.onerror = (error) => {
        reject(new Error(`Failed to load executor_runner.js: ${error}`));
      };
      document.head.appendChild(script);
    });

    return this.initPromise;
  }

  /**
   * Load a model from bytes
   * @param {Uint8Array} modelBytes - Model file bytes (.pte format)
   * @returns {Promise<{modelId: number, inputShapes: Array, outputShapes: Array}>}
   */
  async loadModel(modelBytes) {
    if (!this.isInitialized) {
      throw new Error('ExecuTorchRunner not initialized. Call initialize() first.');
    }

    try {
      const modelId = this.nextModelId++;
      const modelPath = `/models/model_${modelId}.pte`;

      // Create /models directory in virtual filesystem if it doesn't exist
      try {
        this.module.FS.mkdir('/models');
      } catch (e) {
        // Directory already exists, ignore
      }

      // Write model bytes to virtual filesystem
      this.module.FS.writeFile(modelPath, modelBytes);
      this._log(`Wrote model to virtual FS: ${modelPath} (${modelBytes.length} bytes)`);

      // TODO: Actually load the model using ExecuTorch C++ API via Emscripten bindings
      // For now, we just store the path and return metadata
      // In a complete implementation, this would call:
      // this.module._loadModel(modelPath) or similar C++ function

      this.loadedModels.set(modelId, {
        path: modelPath,
        metadata: {
          // TODO: Extract actual shapes from loaded model
          // For now, return empty arrays (will be populated when C++ bindings are added)
          inputShapes: [],
          outputShapes: [],
        }
      });

      this._log(`Model ${modelId} loaded successfully`);

      return {
        modelId: modelId,
        inputShapes: [],
        outputShapes: [],
      };

    } catch (error) {
      this._logError('Failed to load model:', error);
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }

  /**
   * Run inference on a loaded model
   * @param {number} modelId - Model ID from loadModel()
   * @param {Array} inputs - Array of input tensors, each tensor is:
   *   {
   *     shape: Array<number>,
   *     dataType: string ('float32', 'int32', 'int8', 'uint8'),
   *     data: Uint8Array (raw bytes),
   *     name: string (optional)
   *   }
   * @returns {Promise<Array>} Array of output tensors (same format as inputs)
   */
  async forward(modelId, inputs) {
    if (!this.isInitialized) {
      throw new Error('ExecuTorchRunner not initialized. Call initialize() first.');
    }

    if (!this.loadedModels.has(modelId)) {
      throw new Error(`Model ${modelId} not loaded`);
    }

    try {
      this._log(`Running inference on model ${modelId} with ${inputs.length} inputs`);

      // TODO: Actual inference implementation using ExecuTorch C++ API
      // For now, return mock outputs (will be replaced when C++ bindings are added)
      // In a complete implementation, this would:
      // 1. Convert JS tensors to C++ format
      // 2. Call this.module._forward(modelId, inputTensors)
      // 3. Convert C++ output tensors back to JS format

      // Mock output for testing (returns same shape as first input)
      const mockOutputs = inputs.map((input, index) => ({
        shape: input.shape,
        dataType: input.dataType,
        data: new Uint8Array(input.data.length), // Zero-filled for now
        name: `output_${index}`,
      }));

      this._log(`Inference completed, returning ${mockOutputs.length} outputs`);

      return mockOutputs;

    } catch (error) {
      this._logError('Inference failed:', error);
      throw new Error(`Inference failed: ${error.message}`);
    }
  }

  /**
   * Dispose a loaded model and free resources
   * @param {number} modelId - Model ID to dispose
   * @returns {Promise<void>}
   */
  async dispose(modelId) {
    if (!this.isInitialized) {
      return; // Already disposed or never initialized
    }

    if (!this.loadedModels.has(modelId)) {
      this._log(`Model ${modelId} not found (already disposed?)`);
      return;
    }

    try {
      const modelInfo = this.loadedModels.get(modelId);
      const modelPath = modelInfo.path;

      // TODO: Call C++ dispose function
      // this.module._disposeModel(modelId);

      // Remove model file from virtual filesystem
      try {
        this.module.FS.unlink(modelPath);
        this._log(`Removed model file from virtual FS: ${modelPath}`);
      } catch (e) {
        this._log(`Failed to remove model file: ${e.message}`);
      }

      // Remove from loaded models map
      this.loadedModels.delete(modelId);

      this._log(`Model ${modelId} disposed successfully`);

    } catch (error) {
      this._logError('Failed to dispose model:', error);
      throw new Error(`Failed to dispose model: ${error.message}`);
    }
  }

  /**
   * Get metadata for a loaded model
   * @param {number} modelId - Model ID
   * @returns {Object} Model metadata (inputShapes, outputShapes)
   */
  getModelMetadata(modelId) {
    if (!this.loadedModels.has(modelId)) {
      throw new Error(`Model ${modelId} not loaded`);
    }

    return this.loadedModels.get(modelId).metadata;
  }

  /**
   * Check if a model is loaded
   * @param {number} modelId - Model ID
   * @returns {boolean}
   */
  isModelLoaded(modelId) {
    return this.loadedModels.has(modelId);
  }

  /**
   * Get list of loaded model IDs
   * @returns {Array<number>}
   */
  getLoadedModelIds() {
    return Array.from(this.loadedModels.keys());
  }
}

// Create global instance
window.ExecuTorchRunner = new ExecuTorchRunner();

// Export for module systems if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ExecuTorchRunner;
}

// Always log wrapper loaded (not affected by debug flag)
console.log('[ExecuTorch] JavaScript wrapper loaded');
