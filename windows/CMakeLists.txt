# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "executorch_flutter")
project(${PROJECT_NAME} LANGUAGES CXX)

# This plugin uses Native Assets (hook/build.dart) for building native code.
# The native library is built and linked automatically by the Dart Native Assets
# build system, so no bundled libraries are needed here.
set(executorch_flutter_bundled_libraries
  ""
  PARENT_SCOPE
)
