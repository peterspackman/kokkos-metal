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

// Kokkos::abort in a kernel: flags the abort, which the next fence reports,
// and ends the thread.
[[noreturn]] void __kmetal_abort() noexcept;

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

}  // namespace Kokkos::Impl::MetalDevice

#endif
