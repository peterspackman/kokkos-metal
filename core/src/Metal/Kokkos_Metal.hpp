// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_METAL_HPP
#define KOKKOS_METAL_HPP

#include <Kokkos_Concepts.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Metal/Kokkos_MetalSpace.hpp>
#include <Metal/Kokkos_Metal_Intrinsics.hpp>
#include <Metal/Kokkos_Metal_Runtime.hpp>
#include <impl/Kokkos_InitializationSettings.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <type_traits>

namespace Kokkos::Experimental {

// The GPU of an Apple-silicon Mac. One command queue serves every instance:
// kernels run in the order they were enqueued.
class Metal {
 public:
  using execution_space      = Metal;
  using memory_space         = MetalSpace;
  using device_type          = Kokkos::Device<execution_space, memory_space>;
  using array_layout         = LayoutLeft;
  using size_type            = memory_space::size_type;
  using index_type           = std::make_signed_t<size_type>;
  using scratch_memory_space = ScratchMemorySpace<Metal>;

  Metal();

  static void impl_initialize(InitializationSettings const& settings);
  static void impl_finalize();
  static bool impl_is_initialized();

  void print_configuration(std::ostream& os, bool verbose = false) const;
  void fence(std::string const& name =
                 "Kokkos::Metal::fence(): Unnamed Instance Fence") const;
  static void impl_static_fence(std::string const& name);

  static constexpr const char* name() { return "Metal"; }
  // Threads the GPU keeps in flight: what UniqueToken and friends size for.
  static int concurrency();
  uint32_t impl_instance_id() const noexcept { return 1; }

  friend bool operator==(Metal const&, Metal const&) { return true; }
  friend bool operator!=(Metal const&, Metal const&) { return false; }
};

// The vendor GEMM, as cuBLAS is for CUDA: C = alpha op(A) op(B) + beta C,
// plus bias[j] on every row when given, on row-major float matrices in Metal
// memory (leading dimensions in elements), enqueued in order with the kernels.
inline void metal_gemm(bool trans_a, bool trans_b, int m, int n, int k,
                       float alpha, const float* a, int lda, const float* b,
                       int ldb, float beta, float* c, int ldc,
                       const float* bias = nullptr) {
  Kokkos::Impl::MetalRuntime::gemm(trans_a, trans_b, m, n, k, alpha, a, lda, b,
                                   ldb, beta, c, ldc, bias);
}

}  // namespace Kokkos::Experimental

namespace Kokkos::Impl {

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MetalSpace,
                         Kokkos::Experimental::Metal::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
};

// Device memory for a kernel's own bookkeeping (reduction partials), reused
// from launch to launch; kernels run in order, so one buffer suffices.
void* metal_scratch(size_t bytes);

// Enqueue Driver::metal_run() over `groups` threadgroups of `threads`.
template <class Driver>
void metal_launch(const Driver& driver, uint32_t groups, uint32_t threads,
                  size_t threadgroup_bytes = 0) {
  const void* key     = MetalDevice::Kernel<Driver>::instantiate();
  const uint32_t g[3] = {groups, 1, 1}, t[3] = {threads, 1, 1};
  MetalRuntime::dispatch(key, &driver, sizeof(Driver), g, t, threadgroup_bytes);
}

// The most threads a threadgroup of Driver's kernel can have.
template <class Driver>
uint32_t metal_max_threads() {
  return MetalRuntime::max_threads_per_threadgroup(
      MetalDevice::Kernel<Driver>::instantiate());
}

}  // namespace Kokkos::Impl

template <>
struct Kokkos::Tools::Experimental::DeviceTypeTraits<
    ::Kokkos::Experimental::Metal> {
  static constexpr DeviceType id =
      ::Kokkos::Profiling::Experimental::DeviceType::Metal;
  static int device_id(const Kokkos::Experimental::Metal&) { return 0; }
};

#endif
