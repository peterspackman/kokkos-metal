// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_DEEP_COPY_HPP
#define KOKKOS_METAL_DEEP_COPY_HPP

#include <Metal/Kokkos_Metal.hpp>
#include <Metal/Kokkos_MetalSpace.hpp>

#include <string>
#include <type_traits>

namespace Kokkos::Impl {

// Between two Metal allocations: a copy kernel, ordered with the others.
// With host memory involved: wait for the GPU, then copy on the host, which
// shares the memory.
void metal_deep_copy(void* dst, const void* src, size_t n, bool host_involved);

template <class Dst, class Src, class ExecutionSpace>
struct DeepCopy<
    Dst, Src, ExecutionSpace,
    std::enable_if_t<
        (is_metal_type_space<Dst>::value || is_metal_type_space<Src>::value) &&
        (is_metal_type_space<Dst>::value || std::is_same_v<Dst, HostSpace>) &&
        (is_metal_type_space<Src>::value || std::is_same_v<Src, HostSpace>)>> {
  static constexpr bool host_involved =
      !is_metal_type_space<Dst>::value || !is_metal_type_space<Src>::value;

  DeepCopy(void* dst, const void* src, size_t n) {
    metal_deep_copy(dst, src, n, host_involved);
    if constexpr (!host_involved)
      Kokkos::Experimental::Metal().fence(
          "Kokkos::Impl::DeepCopy: fence after a synchronous copy");
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    if constexpr (!std::is_same_v<ExecutionSpace, Kokkos::Experimental::Metal>)
      exec.fence("Kokkos::Impl::DeepCopy: fence before a Metal copy");
    metal_deep_copy(dst, src, n, host_involved);
  }
};

}  // namespace Kokkos::Impl

#endif
