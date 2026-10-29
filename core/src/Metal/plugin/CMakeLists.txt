# The kokkos-metal compiler plugin, a project of its own that Kokkos builds
# when it configures (core/src/CMakeLists.txt). It is built against the LLVM
# of the clang that loads it, so CMAKE_CXX_COMPILER must be that clang:
# Homebrew's llvm on macOS, whose LLVM is found next to it.
cmake_minimum_required(VERSION 3.21)
project(KokkosMetalPlugin LANGUAGES C CXX)

get_filename_component(_kmetal_cxx "${CMAKE_CXX_COMPILER}" REALPATH)
get_filename_component(_kmetal_prefix "${_kmetal_cxx}" DIRECTORY)
find_package(LLVM REQUIRED CONFIG HINTS "${_kmetal_prefix}/../lib/cmake/llvm")

string(REGEX MATCH "^[0-9]+" _kmetal_cxx_major "${CMAKE_CXX_COMPILER_VERSION}")
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT _kmetal_cxx_major STREQUAL LLVM_VERSION_MAJOR)
  message(FATAL_ERROR "The Metal backend needs clang ${LLVM_VERSION_MAJOR} (the clang of LLVM at "
                      "${LLVM_DIR}); CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

add_library(KokkosMetalPlugin MODULE KokkosMetalPlugin.cpp MSLEmitter.cpp)
target_include_directories(KokkosMetalPlugin SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
separate_arguments(_kmetal_llvm_defs NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(KokkosMetalPlugin PRIVATE ${_kmetal_llvm_defs})
# LLVM is built without RTTI; the plugin's own exceptions never cross it.
target_compile_options(KokkosMetalPlugin PRIVATE -fno-rtti -fexceptions)
# LLVM's symbols come from the clang process that loads the plugin.
target_link_options(KokkosMetalPlugin PRIVATE -undefined dynamic_lookup)
set_target_properties(
  KokkosMetalPlugin PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON LIBRARY_OUTPUT_DIRECTORY
                               ${CMAKE_BINARY_DIR}/lib PREFIX lib SUFFIX .dylib
)
