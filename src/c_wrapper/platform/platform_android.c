/**
 * @file platform_android.c
 * @brief Android platform-specific implementations
 */

#ifdef __ANDROID__

#include "platform_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

int et_flutter_platform_file_exists(const char* path) {
    if (!path) {
        return 0;
    }

    struct stat st;
    return stat(path, &st) == 0;
}

int et_flutter_platform_file_readable(const char* path) {
    if (!path) {
        return 0;
    }

    return access(path, R_OK) == 0;
}

size_t et_flutter_platform_file_size(const char* path) {
    if (!path) {
        return 0;
    }

    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }

    return (size_t)st.st_size;
}

char* et_flutter_platform_create_temp_file(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        return NULL;
    }

    // Create temp file in /data/local/tmp or app cache
    char* temp_path = (char*)malloc(256);
    if (!temp_path) {
        return NULL;
    }

    // Use tmpnam to generate unique name
    char template[] = "/data/local/tmp/et_flutter_XXXXXX";
    int fd = mkstemp(template);
    if (fd < 0) {
        free(temp_path);
        return NULL;
    }

    // Write data to file
    ssize_t written = write(fd, data, size);
    close(fd);

    if (written != (ssize_t)size) {
        unlink(template);
        free(temp_path);
        return NULL;
    }

    strcpy(temp_path, template);
    return temp_path;
}

void et_flutter_platform_delete_temp_file(const char* path) {
    if (path) {
        unlink(path);
    }
}

const char* et_flutter_platform_name(void) {
    return "Android";
}

#endif /* __ANDROID__ */
