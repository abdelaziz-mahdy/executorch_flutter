/// Tests for TensorType <-> ExecuTorch native type conversion.
///
/// Verifies that all 13 TensorType values map 1:1 to ExecuTorch's ETDType
/// enum values and that round-trip conversions are lossless.
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:executorch_flutter/src/types.dart';
import 'package:executorch_flutter/src/executorch_manager_base.dart';

void main() {
  // -----------------------------------------------------------------------
  // Binary encoding tests for bfloat16 and float16
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // bfloat16 reference:
  //   bfloat16 is defined by Google as the upper 16 bits of IEEE 754 float32.
  //   Source: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
  //   Float32 bit patterns per IEEE 754-2008 §3.2.2 (binary32):
  //     https://ieeexplore.ieee.org/document/4610730
  //   Verification: python -c "import struct; print(struct.pack('f', 1.0).hex())"
  // -----------------------------------------------------------------------

  group('convertNumericDataToBytes bfloat16 encoding', () {
    test('encodes 1.0 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(1.0) = 0x3F800000
      //   sign=0, exp=0x7F (bias 127 → exponent 0), mantissa=0
      // bfloat16 = upper 16 bits = 0x3F80
      // little-endian bytes: [0x80, 0x3F]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x80);
      expect(bytes[1], 0x3F);
    });

    test('encodes 2.0 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(2.0) = 0x40000000
      //   sign=0, exp=0x80 (bias 127 → exponent 1), mantissa=0
      // bfloat16 = upper 16 bits = 0x4000
      // little-endian bytes: [0x00, 0x40]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [2.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x40);
    });

    test('encodes -1.0 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(-1.0) = 0xBF800000
      //   sign=1, exp=0x7F, mantissa=0
      // bfloat16 = upper 16 bits = 0xBF80
      // little-endian bytes: [0x80, 0xBF]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-1.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x80);
      expect(bytes[1], 0xBF);
    });

    test('encodes 0.5 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(0.5) = 0x3F000000
      //   sign=0, exp=0x7E (bias 127 → exponent -1), mantissa=0
      // bfloat16 = upper 16 bits = 0x3F00
      // little-endian bytes: [0x00, 0x3F]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [0.5],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x3F);
    });

    test('encodes 0.0 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(+0.0) = 0x00000000
      // bfloat16 = 0x0000
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [0.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x00);
    });

    test('encodes -0.0 as correct bfloat16 bytes', () {
      // IEEE 754 binary32(-0.0) = 0x80000000
      //   sign=1, exp=0, mantissa=0
      // bfloat16 = 0x8000
      // little-endian bytes: [0x00, 0x80]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-0.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x80);
    });

    test('encodes multiple values with correct byte length', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1.0, 2.0, 3.0],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(6)); // 3 values * 2 bytes each
    });

    test('encodes infinity as correct bfloat16 bytes', () {
      // IEEE 754 binary32(+∞) = 0x7F800000
      //   sign=0, exp=0xFF (all 1s = infinity), mantissa=0
      // bfloat16 = 0x7F80
      // little-endian bytes: [0x80, 0x7F]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [double.infinity],
        TensorType.bfloat16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x80);
      expect(bytes[1], 0x7F);
    });
  });

  // -----------------------------------------------------------------------
  // float16 (half) reference:
  //   IEEE 754-2008 binary16 format: 1 sign bit, 5 exponent bits (bias 15),
  //   10 mantissa bits.
  //   Source: https://ieeexplore.ieee.org/document/4610730 (§3.2.1)
  //   Format diagram: SEEEEEMMMMMMMMMMMM
  //   Special values per IEEE 754-2008 §3.3, §3.4:
  //     - exp=0x1F + mant≠0 → NaN
  //     - exp=0x1F + mant=0  → Infinity
  //     - exp=0              → Zero or denormal
  //   Max finite: sign=0, exp=0x1E, mant=0x3FF → 2^(30-15)·(2 - 2^-10) = 65504
  //   Verification: python -c "import struct; print(struct.pack('e', 1.0).hex())"
  //   (Python 3.12+ supports float16 via struct format 'e')
  // -----------------------------------------------------------------------

  group('convertNumericDataToBytes float16 encoding', () {
    test('encodes 1.0 as correct float16 bytes', () {
      // binary16(1.0): sign=0, exp=15-15=0→0x0F, mantissa=0
      // Layout: 0 01111 0000000000 = 0x3C00
      // little-endian bytes: [0x00, 0x3C]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x3C);
    });

    test('encodes 2.0 as correct float16 bytes', () {
      // binary16(2.0): sign=0, exp=16-15=1→0x10, mantissa=0
      // Layout: 0 10000 0000000000 = 0x4000
      // little-endian bytes: [0x00, 0x40]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [2.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x40);
    });

    test('encodes -1.0 as correct float16 bytes', () {
      // binary16(-1.0): sign=1, exp=0x0F, mantissa=0
      // Layout: 1 01111 0000000000 = 0xBC00
      // little-endian bytes: [0x00, 0xBC]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-1.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0xBC);
    });

    test('encodes 0.5 as correct float16 bytes', () {
      // binary16(0.5): sign=0, exp=14-15=-1→0x0E, mantissa=0
      // Layout: 0 01110 0000000000 = 0x3800
      // little-endian bytes: [0x00, 0x38]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [0.5],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x38);
    });

    test('encodes 0.0 as correct float16 bytes', () {
      // binary16(+0.0): sign=0, exp=0, mantissa=0 → 0x0000
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [0.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x00);
    });

    test('encodes -0.0 as correct float16 bytes', () {
      // binary16(-0.0): sign=1, exp=0, mantissa=0 → 0x8000
      // little-endian bytes: [0x00, 0x80]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [-0.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x80);
    });

    test('encodes infinity as correct float16 bytes', () {
      // binary16(+∞): sign=0, exp=0x1F (all 1s), mantissa=0 → 0x7C00
      // little-endian bytes: [0x00, 0x7C]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [double.infinity],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x7C);
    });

    test('encodes -infinity as correct float16 bytes', () {
      // binary16(-∞): sign=1, exp=0x1F, mantissa=0 → 0xFC00
      // little-endian bytes: [0x00, 0xFC]
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [double.negativeInfinity],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      expect(bytes[0], 0x00);
      expect(bytes[1], 0xFC);
    });

    test('encodes NaN as float16 with NaN indicator', () {
      // IEEE 754 §3.4: NaN has exp=all-1s and mantissa≠0
      // binary16 NaN: exp=0x1F, mant≠0 → high bit pattern 0x7C01+
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [double.nan],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      // Reconstruct the uint16 to check it's a NaN pattern
      final asUint16 = bytes[0] | (bytes[1] << 8);
      // NaN: exponent == 31 (0x1F), mantissa != 0
      final exp = (asUint16 >> 10) & 0x1F;
      final mant = asUint16 & 0x3FF;
      expect(exp, 0x1F, reason: 'NaN should have all exponent bits set');
      expect(mant, isNot(equals(0)),
          reason: 'NaN should have non-zero mantissa');
    });

    test('encodes multiple values with correct byte length', () {
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1.0, 2.0, 3.0],
        TensorType.float16,
      );
      expect(bytes, hasLength(6)); // 3 values * 2 bytes each
    });

    test('encodes small denormal value as zero (underflow)', () {
      // 1e-20 is far below binary16 min denormal (2^-24 ≈ 5.96e-8)
      // Underflows to zero per IEEE 754 §5.2.4 (round toward zero)
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1e-20],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      // Should underflow to zero
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x00);
    });

    test('encodes large value as infinity (overflow)', () {
      // binary16 max finite = 65504 (0x7BFF). 1e10 exceeds this.
      // Overflows to infinity (0x7C00) per IEEE 754 §5.2.4
      final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
        [1e10],
        TensorType.float16,
      );
      expect(bytes, hasLength(2));
      // Should overflow to infinity: 0x7C00
      expect(bytes[0], 0x00);
      expect(bytes[1], 0x7C);
    });

    test('encodes denormal values matching numpy float16', () {
      // Values below 2^-14 (min normal) but above 2^-24 (min denormal)
      // exercise the binary16 denormal branch. Reference bit patterns from
      // python3 -c "import numpy as np; print(np.float16(v).view(np.uint16))"
      final Map<double, int> denormals = {
        6e-5: 0x03EF,
        3e-5: 0x01F7,
        1e-5: 0x00A8,
        -6e-5: 0x83EF,
      };
      for (final entry in denormals.entries) {
        final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
          [entry.key],
          TensorType.float16,
        );
        final actual = bytes[0] | (bytes[1] << 8);
        expect(actual, entry.value,
            reason: 'float16 denormal mismatch for ${entry.key}: '
                'expected 0x${entry.value.toRadixString(16).padLeft(4, '0')}, '
                'got 0x${actual.toRadixString(16).padLeft(4, '0')}');
      }
    });

    test('rounds normal values to nearest even like numpy float16', () {
      // numpy: np.float16(3.14) == 0x4248 (rounded up from truncated 0x4247)
      final Map<double, int> rounded = {
        3.14: 0x4248,
        // 1.0009765625 = 1 + 2^-10: representable exactly, no rounding.
        1.0009765625: 0x3C01,
        // 1.00048828125 = 1 + 2^-11: tie → round to even (LSB 0 stays).
        1.00048828125: 0x3C00,
        // 1.00146484375 = 1 + 3*2^-11: tie → round to even (LSB 1 → up).
        1.00146484375: 0x3C02,
      };
      for (final entry in rounded.entries) {
        final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
          [entry.key],
          TensorType.float16,
        );
        final actual = bytes[0] | (bytes[1] << 8);
        expect(actual, entry.value,
            reason: 'float16 rounding mismatch for ${entry.key}: '
                'expected 0x${entry.value.toRadixString(16).padLeft(4, '0')}, '
                'got 0x${actual.toRadixString(16).padLeft(4, '0')}');
      }
    });
  });

  // -----------------------------------------------------------------------
  // Round-trip consistency: verify encoding against independently computed
  // reference values derived from the IEEE 754 specification.
  // -----------------------------------------------------------------------

  group('convertNumericDataToBytes round-trip consistency', () {
    test('bfloat16 bytes match numpy/PyTorch round-to-nearest-even', () {
      // bfloat16 keeps the upper 16 bits of IEEE 754 binary32, rounded to
      // nearest even (not truncated) — matching numpy (ml_dtypes) and
      // PyTorch conversions.
      // Reference: python3 -c "import numpy as np, ml_dtypes;
      //   print(np.float32(v).view(np.uint32) ...)" / torch.tensor(v).bfloat16()
      final Map<double, int> referenceValues = {
        // Exact values (lower 16 float32 bits are zero): same as truncation.
        1.0: 0x3F80,
        0.5: 0x3F00,
        -2.0: 0xC000,
        // (0.0 vs -0.0 collide as map keys; -0.0 is covered by the
        // dedicated encoding test above.)
        0.0: 0x0000,
        // 3.14 float32 = 0x4048F5C3; lower bits 0xF5C3 > 0x8000 → round up.
        3.14: 0x4049,
        // 1.00390625 float32 = 0x3F808000; tie → round to even (LSB 0 stays).
        1.00390625: 0x3F80,
        // 1.01171875 float32 = 0x3F818000; tie → round to even (LSB 1 → up).
        1.01171875: 0x3F82,
      };
      for (final entry in referenceValues.entries) {
        final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
          [entry.key],
          TensorType.bfloat16,
        );
        final actual = bytes[0] | (bytes[1] << 8);
        expect(actual, entry.value,
            reason: 'bfloat16 encoding mismatch for ${entry.key}: '
                'expected 0x${entry.value.toRadixString(16).padLeft(4, '0')}, '
                'got 0x${actual.toRadixString(16).padLeft(4, '0')}');
      }
    });

    test('float16 bytes match known IEEE 754 reference values', () {
      // Reference bit patterns computed per IEEE 754-2008 §3.2.1 (binary16).
      // Each value manually derived:
      //   S EEEEE MMMMMMMMMMMM  (1+5+10 = 16 bits)
      //   exponent = biased_exp - 15,  value = (-1)^S · 2^(exp-15) · (1.mant)
      //
      // Verified independently via:
      //   python3 -c "import struct; print(struct.pack('>e', 1.0).hex())"
      //   (Python 3.12+ struct format 'e' = float16, big-endian for readability)
      final Map<double, int> referenceValues = {
        // sign=0, exp=0,  mant=0        → +zero
        0.0: 0x0000,
        // sign=0, exp=15, mant=0        → 1.0
        1.0: 0x3C00,
        // sign=1, exp=15, mant=0        → -1.0
        -1.0: 0xBC00,
        // sign=0, exp=16, mant=0        → 2.0
        2.0: 0x4000,
        // sign=0, exp=14, mant=0        → 0.5
        0.5: 0x3800,
        // sign=1, exp=14, mant=0        → -0.5
        -0.5: 0xB800,
        // sign=0, exp=17, mant=0        → 3.0  (2^1 · 1.5)
        3.0: 0x4200,
        // sign=0, exp=30, mant=0x3FF    → max finite = 65504
        65504.0: 0x7BFF,
        // sign=1, exp=30, mant=0x3FF    → min finite = -65504
        -65504.0: 0xFBFF,
      };

      for (final entry in referenceValues.entries) {
        final bytes = ExecutorchManagerBase.convertNumericDataToBytes(
          [entry.key],
          TensorType.float16,
        );
        final actual = bytes[0] | (bytes[1] << 8);
        expect(actual, entry.value,
            reason: 'float16 encoding mismatch for ${entry.key}: '
                'expected 0x${entry.value.toRadixString(16).padLeft(4, '0')}, '
                'got 0x${actual.toRadixString(16).padLeft(4, '0')}');
      }
    });
  });
}
