// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// The host side of the Metal backend: allocation, kernel registry, dispatch,
// synchronisation. Plain C++ over an Objective-C++ implementation, and free of
// Kokkos headers so that it can be tested on its own.

#ifndef KOKKOS_METAL_RUNTIME_HPP
#define KOKKOS_METAL_RUNTIME_HPP

#include <cstddef>
#include <cstdint>

namespace Kokkos::Impl::MetalRuntime {

// One kernel emitted by the compiler plugin. `prelude` is shared by all the
// kernels of a translation unit; a kernel's full MSL is prelude + source.
struct KernelDesc {
  const void* key;
  const char* name;
  const char* prelude;
  const char* source;
  uint32_t flags;
};

enum KernelFlags : uint32_t {
  kUsesThreadgroupMemory = 1u << 0,
  kUsesAbort             = 1u << 1,
  // The plugin could not translate it; `source` says why.
  kUnsupported = 1u << 2,
  kUsesPrintf  = 1u << 3,
};

void initialize();
void finalize();
bool is_initialized();

// Memory every kernel can read and write, at the same address on the host
// and the GPU. Freeing fences first: a kernel in flight may still use it.
void* allocate(size_t bytes);
void deallocate(void* ptr);

// Enqueues kernel `key` (MetalDevice::Kernel<Driver>::key) over `groups`
// threadgroups of `threads` threads. The `args_size` bytes at `args` are
// copied before return. `threadgroup_bytes` is the [[threadgroup(0)]] size.
void dispatch(const void* key, const void* args, size_t args_size,
              const uint32_t groups[3], const uint32_t threads[3],
              size_t threadgroup_bytes);

// Largest threadgroup the compiled kernel allows and its SIMD width. Compiles
// the kernel if it was not yet.
uint32_t max_threads_per_threadgroup(const void* key);
uint32_t simd_width(const void* key);
size_t max_threadgroup_memory();

// Asynchronous, ordered with kernels: dst and src are Metal allocations.
void copy(void* dst, const void* src, size_t bytes);
void fill_zero(void* dst, size_t bytes);

// C = alpha op(A) op(B) + beta C (+ bias[j] on every row) on row-major float
// matrices in Metal memory, ordered with the kernels.
void gemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
          const float* a, int lda, const float* b, int ldb, float beta,
          float* c, int ldc, const float* bias = nullptr);

// Waits until everything enqueued so far has completed.
void fence();

const char* device_name();

}  // namespace Kokkos::Impl::MetalRuntime

// Called from the static constructor the plugin adds to each translation unit.
extern "C" void kokkos_metal_register_kernels(
    const Kokkos::Impl::MetalRuntime::KernelDesc* kernels, uint32_t count);

#endif
