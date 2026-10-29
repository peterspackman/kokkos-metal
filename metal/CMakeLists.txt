# Tests of the Metal backend: this Kokkos built with Serial and Metal, and
# programs checked against the host.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
#   cmake --build build-metal && ctest --test-dir build-metal
cmake_minimum_required(VERSION 3.21)
project(kokkos_metal_tests LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_METAL ON CACHE BOOL "")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. kokkos)

enable_testing()
# test_runtime: the plugin and runtime on their own, without Kokkos's
# parallel patterns (including fp64 against the CPU and the GEMM).
foreach(name test_runtime test_range test_team)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE Kokkos::kokkos)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
