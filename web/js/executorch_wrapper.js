/**
 * ExecuTorch Wasm JavaScript Wrapper
 *
 * Provides a clean JavaScript API around the Emscripten Embind-generated Module
 * for loading models, running inference, and managing resources.
 *
 * This wrapper uses the official ExecuTorch Wasm bindings which expose:
 * - Module.load(data) - Load model from Uint8Array/ArrayBuffer
 * - module.forward(inputs) - Run forward pass
 * - Tensor.fromArray(sizes, data, type) - Create tensors
 *
 * This wrapper is designed to be called from Dart via dart:js_interop.
 */

class ExecuTorchRunner {
  constructor() {
    this.wasmModule = null;
    this.isInitialized = false;
    this.loadedModels = new Map(); // modelId -> JsModule instance
    this.nextModelId = 0;
    this.initPromise = null;
    this.debugLoggingEnabled = false;
  }

  /**
   * Enable or disable debug logging
   * @param {boolean} enabled - true to enable, false to disable
   */
  setDebugLogging(enabled) {
    this.debugLoggingEnabled = enabled;
    if (enabled) {
      console.log('[ExecuTorch] Debug logging enabled');
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
      // Load the modular executorch.js which exports createExecuTorchModule
      const script = document.createElement('script');
      script.src = 'wasm/executorch.js';
      script.async = true;

      script.onload = async () => {
        this._log('executorch.js loaded, initializing module...');

        try {
          // createExecuTorchModule is the factory function from MODULARIZE
          if (typeof createExecuTorchModule !== 'function') {
            throw new Error('createExecuTorchModule not found. Make sure executorch.js is built with -sMODULARIZE=1');
          }

          // Initialize the Wasm module
          this.wasmModule = await createExecuTorchModule();
          this.isInitialized = true;
          this._log('Wasm module initialized successfully');

          // Log available exports for debugging
          if (this.debugLoggingEnabled) {
            this._log('Available exports:', Object.keys(this.wasmModule));
          }

          resolve();
        } catch (error) {
          this._logError('Failed to initialize Wasm module:', error);
          reject(new Error(`Failed to initialize ExecuTorch Wasm: ${error.message}`));
        }
      };

      script.onerror = (error) => {
        this._logError('Failed to load executorch.js:', error);
        reject(new Error('Failed to load executorch.js. Make sure the file exists at wasm/executorch.js'));
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
      this._log(`Loading model ${modelId} (${modelBytes.length} bytes)...`);

      // Use the Embind Module.load() API
      // It accepts Uint8Array directly
      const jsModule = this.wasmModule.Module.load(modelBytes);

      // Get method metadata for shapes
      let inputShapes = [];
      let outputShapes = [];

      try {
        // Get available methods
        const methods = jsModule.getMethods();
        this._log(`Model methods: ${JSON.stringify(methods)}`);

        // Get metadata for 'forward' method if available
        if (methods.includes('forward')) {
          jsModule.loadMethod('forward');
          const meta = jsModule.getMethodMeta('forward');

          // Extract input shapes
          if (meta.inputTensorMeta) {
            inputShapes = meta.inputTensorMeta
              .filter(t => t !== undefined)
              .map(t => Array.from(t.sizes));
          }

          // Extract output shapes
          if (meta.outputTensorMeta) {
            outputShapes = meta.outputTensorMeta
              .filter(t => t !== undefined)
              .map(t => Array.from(t.sizes));
          }
        }
      } catch (metaError) {
        this._log(`Could not get method metadata: ${metaError.message}`);
      }

      // Store the module
      this.loadedModels.set(modelId, {
        module: jsModule,
        inputShapes: inputShapes,
        outputShapes: outputShapes,
      });

      this._log(`Model ${modelId} loaded successfully`);
      this._log(`Input shapes: ${JSON.stringify(inputShapes)}`);
      this._log(`Output shapes: ${JSON.stringify(outputShapes)}`);

      return {
        modelId: modelId,
        inputShapes: inputShapes,
        outputShapes: outputShapes,
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
   *     dataType: string ('float32', 'int64'),
   *     data: Uint8Array (raw bytes) or Float32Array/BigInt64Array,
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

    const modelInfo = this.loadedModels.get(modelId);
    const jsModule = modelInfo.module;

    try {
      this._log(`Running inference on model ${modelId} with ${inputs.length} inputs`);

      // Convert input tensors to Embind Tensor objects
      const embindInputs = inputs.map((input, idx) => {
        return this._createEmbindTensor(input);
      });

      // Run forward pass
      const outputs = jsModule.forward(embindInputs);

      // Convert output Embind Tensors back to our format
      const result = [];
      for (let i = 0; i < outputs.length; i++) {
        const tensor = outputs[i];
        result.push(this._embindTensorToOutput(tensor, `output_${i}`));
      }

      this._log(`Inference completed, returning ${result.length} outputs`);

      return result;

    } catch (error) {
      this._logError('Inference failed:', error);
      throw new Error(`Inference failed: ${error.message}`);
    }
  }

  /**
   * Create an Embind Tensor from our input format
   * @private
   */
  _createEmbindTensor(input) {
    const { shape, dataType, data } = input;
    const Tensor = this.wasmModule.Tensor;
    const ScalarType = this.wasmModule.ScalarType;

    // Convert raw bytes to typed array based on dataType
    let typedData;
    let scalarType;

    switch (dataType) {
      case 'float32':
        typedData = data instanceof Float32Array
          ? Array.from(data)
          : Array.from(new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4));
        scalarType = ScalarType.Float;
        break;

      case 'int64':
        // BigInt64Array for int64
        typedData = data instanceof BigInt64Array
          ? Array.from(data)
          : Array.from(new BigInt64Array(data.buffer, data.byteOffset, data.byteLength / 8));
        scalarType = ScalarType.Long;
        break;

      default:
        throw new Error(`Unsupported tensor type: ${dataType}. Supported: float32, int64`);
    }

    // Create tensor using Embind API
    return Tensor.fromArray(shape, typedData, scalarType);
  }

  /**
   * Convert an Embind Tensor to our output format
   * @private
   */
  _embindTensorToOutput(tensor, name) {
    const scalarType = tensor.scalarType;
    const sizes = Array.from(tensor.sizes);
    const data = tensor.data; // Returns typed memory view

    // Determine dataType string from ScalarType
    let dataType;
    let outputData;

    const ScalarType = this.wasmModule.ScalarType;

    if (scalarType === ScalarType.Float) {
      dataType = 'float32';
      // Copy the data to a new Uint8Array (data is a Float32Array view)
      outputData = new Uint8Array(new Float32Array(data).buffer);
    } else if (scalarType === ScalarType.Long) {
      dataType = 'int64';
      // Copy the data to a new Uint8Array
      outputData = new Uint8Array(new BigInt64Array(data).buffer);
    } else {
      throw new Error(`Unsupported output tensor type: ${scalarType.name || scalarType}`);
    }

    return {
      shape: sizes,
      dataType: dataType,
      data: outputData,
      name: name,
    };
  }

  /**
   * Dispose a loaded model and free resources
   * @param {number} modelId - Model ID to dispose
   * @returns {Promise<void>}
   */
  async dispose(modelId) {
    if (!this.isInitialized) {
      return;
    }

    if (!this.loadedModels.has(modelId)) {
      this._log(`Model ${modelId} not found (already disposed?)`);
      return;
    }

    try {
      const modelInfo = this.loadedModels.get(modelId);

      // The JsModule will be garbage collected
      // Embind handles cleanup automatically
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

    const modelInfo = this.loadedModels.get(modelId);
    return {
      inputShapes: modelInfo.inputShapes,
      outputShapes: modelInfo.outputShapes,
    };
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
console.log('[ExecuTorch] JavaScript wrapper loaded (Embind version)');
