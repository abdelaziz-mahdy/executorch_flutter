/**
 * @file platform_utils.h
 * @brief Platform-specific utility functions
 *
 * Provides platform-agnostic wrappers for file operations and platform-specific
 * functionality. Platform-specific implementations in platform_android.c,
 * platform_ios.c, and platform_macos.c
 *
 * @date 2025-10-18
 * @version 1.0
 */

#ifndef EXECUTORCH_FLUTTER_PLATFORM_UTILS_H
#define EXECUTORCH_FLUTTER_PLATFORM_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Check if file exists
 *
 * @param path File path (null-terminated UTF-8)
 * @return 1 if file exists, 0 otherwise
 */
int et_flutter_platform_file_exists(const char* path);

/**
 * @brief Check if file is readable
 *
 * @param path File path (null-terminated UTF-8)
 * @return 1 if file is readable, 0 otherwise
 */
int et_flutter_platform_file_readable(const char* path);

/**
 * @brief Get file size in bytes
 *
 * @param path File path (null-terminated UTF-8)
 * @return File size in bytes, or 0 on error
 */
size_t et_flutter_platform_file_size(const char* path);

/**
 * @brief Create temporary file from data
 *
 * Creates a temporary file and writes the provided data to it.
 * Used for loading models from memory (Uint8List from Dart).
 *
 * @param data Data to write
 * @param size Size of data in bytes
 * @return Path to temporary file (caller must free), or NULL on error
 */
char* et_flutter_platform_create_temp_file(const uint8_t* data, size_t size);

/**
 * @brief Delete temporary file
 *
 * @param path Path to file to delete
 */
void et_flutter_platform_delete_temp_file(const char* path);

/**
 * @brief Get platform name (for debugging)
 *
 * @return Platform name ("Android", "iOS", "macOS")
 */
const char* et_flutter_platform_name(void);

#ifdef __cplusplus
}
#endif

#endif /* EXECUTORCH_FLUTTER_PLATFORM_UTILS_H */
