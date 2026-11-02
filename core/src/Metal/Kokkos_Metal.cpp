// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>
#include <Metal/Kokkos_Metal.hpp>
#include <Metal/Kokkos_Metal_Runtime.hpp>
#include <impl/Kokkos_CheckUsage.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

#include <cstring>
#include <mutex>
#include <ostream>

namespace Kokkos {
namespace Experimental {

Metal::Metal() {
  Kokkos::Impl::check_execution_space_constructor_precondition(name());
}

void Metal::impl_initialize(InitializationSettings const&) {
  Kokkos::Impl::MetalRuntime::initialize();
}

void Metal::impl_finalize() {
  (void)Kokkos::Impl::metal_global_unique_token_locks(true);
  Kokkos::Impl::MetalRuntime::finalize();
}

bool Metal::impl_is_initialized() {
  return Kokkos::Impl::MetalRuntime::is_initialized();
}

void Metal::print_configuration(std::ostream& os, bool) const {
  os << "Device Execution Space:\n";
  os << "  KOKKOS_ENABLE_METAL: yes\n";
  os << "\nMetal Runtime Configuration:\n";
  os << "  Device: " << Kokkos::Impl::MetalRuntime::device_name() << "\n";
}

void Metal::fence(std::string const& name) const { impl_static_fence(name); }

void Metal::impl_static_fence(std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Metal>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [] { Kokkos::Impl::MetalRuntime::fence(); });
}


namespace Impl {

template <class Space>
void* MetalSpaceBase<Space>::impl_allocate(const char* label, size_t size,
                                           size_t logical_size,
                                           Kokkos::Tools::SpaceHandle handle) {
  void* ptr = Kokkos::Impl::MetalRuntime::allocate(size);
  if (!ptr && size) Kokkos::Impl::throw_bad_alloc(Space::name(), size, label);
  if (Kokkos::Profiling::profileLibraryLoaded())
    Kokkos::Profiling::allocateData(handle, label, ptr,
                                    logical_size ? logical_size : size);
  return ptr;
}

template <class Space>
void MetalSpaceBase<Space>::impl_deallocate(const char* label, void* ptr,
                                            size_t size, size_t logical_size,
                                            Kokkos::Tools::SpaceHandle handle) {
  if (Kokkos::Profiling::profileLibraryLoaded())
    Kokkos::Profiling::deallocateData(handle, label, ptr,
                                      logical_size ? logical_size : size);
  Kokkos::Impl::MetalRuntime::deallocate(ptr);
}

template class MetalSpaceBase<MetalSpace>;
template class MetalSpaceBase<MetalSharedSpace>;

}  // namespace Impl
}  // namespace Experimental

namespace Impl {

namespace {

struct ScratchPool {
  std::mutex mutex;
  void* ptr   = nullptr;
  size_t size = 0;

  MetalScratch get(size_t bytes) {
    std::unique_lock<std::mutex> lock(mutex);
    if (bytes > size) {
      MetalRuntime::deallocate(ptr);  // fences: a kernel may still use it
      size = std::max(bytes, 2 * size);
      ptr  = MetalRuntime::allocate(size);
    }
    return {std::move(lock), static_cast<char*>(ptr)};
  }
};

}  // namespace

MetalScratch metal_scratch(size_t bytes) {
  static ScratchPool pool;
  return pool.get(bytes);
}

// Apart from metal_scratch, which a team reduction uses at the same time.
MetalScratch metal_team_scratch(size_t bytes) {
  static ScratchPool pool;
  return pool.get(bytes);
}

Kokkos::View<uint32_t*, Kokkos::Experimental::MetalSpace>
metal_global_unique_token_locks(bool deallocate) {
  static Kokkos::View<uint32_t*, Kokkos::Experimental::MetalSpace> locks;
  if (deallocate)
    locks = {};
  else if (locks.extent(0) == 0)
    locks = Kokkos::View<uint32_t*, Kokkos::Experimental::MetalSpace>(
        "Kokkos::UniqueToken<Metal>::m_locks",
        Kokkos::Experimental::Metal::concurrency());
  return locks;
}

void metal_deep_copy(void* dst, const void* src, size_t n, bool host_involved) {
  if (n == 0) return;
  if (host_involved) {
    MetalRuntime::fence();
    std::memcpy(dst, src, n);
  } else {
    MetalRuntime::copy(dst, src, n);
  }
}

int g_metal_space_factory_initialized =
    initialize_space_factory<Kokkos::Experimental::Metal>("150_Metal");

}  // namespace Impl
}  // namespace Kokkos

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::Experimental::MetalSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::Experimental::MetalSharedSpace);
