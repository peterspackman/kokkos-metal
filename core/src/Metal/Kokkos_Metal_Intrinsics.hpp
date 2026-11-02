// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// The contract between the Metal backend's C++ and the kokkos-metal compiler
// plugin. Device code is ordinary C++ compiled by the host compiler; the
// plugin clones every kernel entry (KOKKOS_IMPL_METAL_KERNEL_ENTRY) into a
// device module and translates it to Metal Shading Language, mapping each
// __kmetal_* function below to the Metal builtin it names. In host code the
// plugin folds __kmetal_is_device() to false and makes the others trap.
//
// Free of Kokkos headers on purpose: the runtime and the standalone tests
// include it too.

#ifndef KOKKOS_METAL_INTRINSICS_HPP
#define KOKKOS_METAL_INTRINSICS_HPP

#include <cstdint>
#include <type_traits>

extern "C" {
// [[thread_position_in_grid]] and friends, component d (0, 1 or 2).
uint32_t __kmetal_thread_position_in_grid(uint32_t d) noexcept;
uint32_t __kmetal_thread_position_in_threadgroup(uint32_t d) noexcept;
uint32_t __kmetal_threadgroup_position_in_grid(uint32_t d) noexcept;
uint32_t __kmetal_threads_per_threadgroup(uint32_t d) noexcept;
uint32_t __kmetal_threadgroups_per_grid(uint32_t d) noexcept;
uint32_t __kmetal_thread_index_in_simdgroup() noexcept;
uint32_t __kmetal_simdgroup_index_in_threadgroup() noexcept;
uint32_t __kmetal_threads_per_simdgroup() noexcept;

// Barriers and shuffles are convergent: the optimiser must not make them
// depend on any more conditions than the source does (by unswitching a loop
// on the SIMD lane, say), or lanes would wait on, or read from, lanes that
// are not there.

// threadgroup_barrier(mem_threadgroup | mem_device) and simdgroup_barrier.
__attribute__((convergent)) void __kmetal_threadgroup_barrier() noexcept;
__attribute__((convergent)) void __kmetal_simdgroup_barrier() noexcept;

// The dynamic threadgroup memory, [[threadgroup(0)]].
void* __kmetal_threadgroup_memory() noexcept;

// simd_shuffle / simd_shuffle_xor / simd_shuffle_down on 32-bit words; wider
// types are moved a word at a time.
__attribute__((convergent)) uint32_t
__kmetal_simd_shuffle(uint32_t v, uint32_t lane) noexcept;
__attribute__((convergent)) uint32_t
__kmetal_simd_shuffle_xor(uint32_t v, uint32_t mask) noexcept;
__attribute__((convergent)) uint32_t
__kmetal_simd_shuffle_down(uint32_t v, uint32_t delta) noexcept;
__attribute__((convergent)) uint32_t
__kmetal_simd_shuffle_up(uint32_t v, uint32_t delta) noexcept;

// True in a kernel, false on the host.
__attribute__((const)) bool __kmetal_is_device() noexcept;

// Kokkos::abort in a kernel: flags the abort, which the next fence reports
// with the message of the first thread to abort, and ends the thread.
[[noreturn]] void __kmetal_abort(const char* message) noexcept;

// Kokkos::printf in a kernel appends a record to this buffer: a 32-bit offset
// and a capacity, then the records from byte 16. The host prints them at the
// next fence.
void* __kmetal_printf_buffer() noexcept;

// First statement of each kernel entry: names the key the host launches it by.
void __kmetal_kernel_key(const void* key) noexcept;
}

namespace Kokkos::Impl::MetalDevice {

// A kernel entry. The plugin finds the annotation and makes `entry` a Metal
// kernel taking the bitwise copy of a Driver as its argument buffer; the host
// launches it by &key. Driver::metal_run() is the per-thread body.
template <class Driver>
struct Kernel {
  static inline char key = 0;

  __attribute__((annotate("kokkos_metal_kernel"), used, noinline)) static void
  entry(const Driver* driver) {
    __kmetal_kernel_key(&key);
    driver->metal_run();
  }

  // ODR-uses the entry so that it is instantiated in this translation unit.
  static const void* instantiate() {
    (void)&entry;
    return &key;
  }
};

template <class T>
uint64_t printf_arg(T v) {
  if constexpr (std::is_floating_point_v<T>)
    return __builtin_bit_cast(uint64_t, double(v));
  else if constexpr (std::is_pointer_v<T>)
    return reinterpret_cast<uintptr_t>(v);
  else if constexpr (std::is_signed_v<T>)
    return uint64_t(int64_t(v));
  else
    return uint64_t(v);
}

// A printf record: its size in bytes (a multiple of 8), the argument count,
// the arguments widened to 64 bits (floating point to double), the format.
template <class... Args>
void printf(const char* format, Args... args) {
  constexpr uint32_t n = sizeof...(Args);
  uint32_t len         = 0;
  while (format[len]) ++len;
  const uint64_t values[n + 1] = {printf_arg(args)..., 0};
  const uint32_t size          = (8 + 8 * n + len + 1 + 7) & ~7u;
  auto* buf      = static_cast<char*>(__kmetal_printf_buffer());
  auto* header   = reinterpret_cast<uint32_t*>(buf);
  const uint32_t off = __atomic_fetch_add(header, size, __ATOMIC_RELAXED);
  if (off + size > header[1]) return;
  char* rec                           = buf + 16 + off;
  reinterpret_cast<uint32_t*>(rec)[0] = size;
  reinterpret_cast<uint32_t*>(rec)[1] = n;
  for (uint32_t i = 0; i < n; ++i)
    reinterpret_cast<uint64_t*>(rec + 8)[i] = values[i];
  for (uint32_t i = 0; i <= len; ++i) rec[8 + 8 * n + i] = format[i];
}

}  // namespace Kokkos::Impl::MetalDevice

#endif
