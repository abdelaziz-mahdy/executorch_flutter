/// Tensor conversion utilities for FFI bridge
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import '../generated/executorch_api.dart'; // Pigeon TensorData
import '../generated/executorch_ffi_bindings.dart'; // C structs

/// Converts Dart TensorData to C ETFlutterTensorData (allocates C memory)
///
/// This function allocates unmanaged C memory for the tensor struct and data.
/// The caller is responsible for freeing this memory using [freeTensorData].
///
/// Memory ownership: Dart → C (zero-copy for data pointer, C borrows Dart's data)
ffi.Pointer<ETFlutterTensorData> toCTensor(TensorData dartTensor) {
  // Allocate C struct
  final cTensor = calloc<ETFlutterTensorData>();

  // Convert and set dtype (assign enum value as int)
  cTensor.ref.dtype = _dartTypeToCType(dartTensor.dataType).value;

  // Set shape
  final numDims = dartTensor.shape.length;
  if (numDims > 8) {
    // ET_FLUTTER_MAX_TENSOR_DIMS
    calloc.free(cTensor);
    throw ArgumentError('Too many dimensions: $numDims (max 8)');
  }

  cTensor.ref.shape.num_dims = numDims;
  for (int i = 0; i < numDims; i++) {
    cTensor.ref.shape.dims[i] = dartTensor.shape[i]!;
  }

  // Set data pointer (zero-copy - C borrows Dart's data)
  // Dart's Uint8List data will remain valid as long as dartTensor is in scope
  final dataPtr = malloc.allocate<ffi.Uint8>(dartTensor.data.length);
  final dataList = dataPtr.asTypedList(dartTensor.data.length);
  dataList.setAll(0, dartTensor.data);

  cTensor.ref.data = dataPtr.cast<ffi.Void>();
  cTensor.ref.data_size = dartTensor.data.length;

  // Set name (optional)
  if (dartTensor.name != null && dartTensor.name!.isNotEmpty) {
    final nameUnits = dartTensor.name!.codeUnits;
    final maxLen = 64 - 1; // ET_FLUTTER_TENSOR_NAME_MAX_LEN - 1 for null terminator
    final copyLen = nameUnits.length > maxLen ? maxLen : nameUnits.length;

    for (int i = 0; i < copyLen; i++) {
      cTensor.ref.name[i] = nameUnits[i];
    }
    cTensor.ref.name[copyLen] = 0; // Null terminator
  } else {
    cTensor.ref.name[0] = 0; // Empty string
  }

  return cTensor;
}

/// Converts C ETFlutterTensorData to Dart TensorData (copies data)
///
/// This function creates a new Dart TensorData by copying data from the C struct.
/// The C struct memory remains owned by the caller and must be freed separately.
///
/// Memory ownership: C → Dart (single copy - Dart owns the copy)
TensorData fromCTensor(ffi.Pointer<ETFlutterTensorData> cTensor) {
  if (cTensor.address == 0) {
    throw ArgumentError('Null tensor pointer');
  }

  final ref = cTensor.ref;

  // Extract dtype (convert int to enum first)
  final dataType = _cTypeToDartType(
    ETFlutterDataType.fromValue(ref.dtype),
  );

  // Extract shape
  final shape = List<int?>.filled(ref.shape.num_dims, null);
  for (int i = 0; i < ref.shape.num_dims; i++) {
    shape[i] = ref.shape.dims[i];
  }

  // Copy data from C to Dart (single copy)
  final dataSize = ref.data_size;
  final dataPtr = ref.data.cast<ffi.Uint8>();
  final data = Uint8List(dataSize);

  if (dataPtr.address != 0) {
    final sourceList = dataPtr.asTypedList(dataSize);
    data.setAll(0, sourceList);
  }

  // Extract name (if present)
  String? name;
  if (ref.name[0] != 0) {
    final nameBytes = <int>[];
    for (int i = 0; i < 64 && ref.name[i] != 0; i++) {
      nameBytes.add(ref.name[i]);
    }
    name = String.fromCharCodes(nameBytes);
  }

  return TensorData(
    shape: shape,
    dataType: dataType,
    data: data,
    name: name,
  );
}

/// Frees C tensor data allocated by [toCTensor]
///
/// This frees both the data buffer and the struct itself.
/// Safe to call with null pointer (no-op).
void freeTensorData(ffi.Pointer<ETFlutterTensorData>? cTensor) {
  if (cTensor == null || cTensor.address == 0) {
    return;
  }

  // Free data buffer if allocated
  if (cTensor.ref.data.address != 0) {
    malloc.free(cTensor.ref.data);
  }

  // Free struct
  calloc.free(cTensor);
}

/// Maps Pigeon TensorType to C ETFlutterDataType
ETFlutterDataType _dartTypeToCType(TensorType dartType) {
  switch (dartType) {
    case TensorType.float32:
      return ETFlutterDataType.ET_FLUTTER_DTYPE_FLOAT32;
    case TensorType.int32:
      return ETFlutterDataType.ET_FLUTTER_DTYPE_INT32;
    case TensorType.int8:
      return ETFlutterDataType.ET_FLUTTER_DTYPE_INT8;
    case TensorType.uint8:
      return ETFlutterDataType.ET_FLUTTER_DTYPE_UINT8;
  }
}

/// Maps C ETFlutterDataType to Pigeon TensorType
TensorType _cTypeToDartType(ETFlutterDataType cType) {
  switch (cType) {
    case ETFlutterDataType.ET_FLUTTER_DTYPE_FLOAT32:
      return TensorType.float32;
    case ETFlutterDataType.ET_FLUTTER_DTYPE_INT32:
      return TensorType.int32;
    case ETFlutterDataType.ET_FLUTTER_DTYPE_INT8:
      return TensorType.int8;
    case ETFlutterDataType.ET_FLUTTER_DTYPE_UINT8:
      return TensorType.uint8;
  }
}

/// Converts list of Dart tensors to C array (allocates C memory)
///
/// Returns pointer to C array and the number of tensors.
/// Caller must free using [freeTensorArray].
(ffi.Pointer<ffi.Pointer<ETFlutterTensorData>>, int) toCTensorArray(
  List<TensorData> dartTensors,
) {
  if (dartTensors.isEmpty) {
    return (ffi.nullptr.cast<ffi.Pointer<ETFlutterTensorData>>(), 0);
  }

  // Allocate array of pointers
  final arrayPtr = calloc<ffi.Pointer<ETFlutterTensorData>>(dartTensors.length);

  // Convert each tensor
  for (int i = 0; i < dartTensors.length; i++) {
    arrayPtr[i] = toCTensor(dartTensors[i]);
  }

  return (arrayPtr, dartTensors.length);
}

/// Frees C tensor array allocated by [toCTensorArray]
void freeTensorArray(
  ffi.Pointer<ffi.Pointer<ETFlutterTensorData>> arrayPtr,
  int count,
) {
  if (arrayPtr.address == 0) {
    return;
  }

  // Free each tensor
  for (int i = 0; i < count; i++) {
    freeTensorData(arrayPtr[i]);
  }

  // Free array
  calloc.free(arrayPtr);
}
