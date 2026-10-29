// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_METAL_SPACE_HPP
#define KOKKOS_METAL_SPACE_HPP

#include <Kokkos_Concepts.hpp>
#include <Kokkos_HostSpace.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Tools.hpp>

#include <cstddef>
#include <type_traits>

// Both spaces allocate the same memory: Apple GPUs share it with the CPU, and
// the runtime maps every allocation at one address for both. MetalSpace is
// nevertheless host-inaccessible as far as Kokkos is concerned, so that
// mirrors are separate host copies and a host access cannot race a kernel;
// MetalSharedSpace is the one to use for memory both sides touch.

namespace Kokkos::Experimental {

namespace Impl {
template <class Space>
class MetalSpaceBase {
 public:
  using memory_space    = Space;
  using execution_space = Metal;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = size_t;

  void* allocate(const size_t arg_alloc_size) const {
    return allocate("[unlabeled]", arg_alloc_size);
  }
  template <class ExecutionSpace>
  void* allocate(const ExecutionSpace&, const size_t arg_alloc_size) const {
    return allocate(arg_alloc_size);
  }
  template <class ExecutionSpace>
  void* allocate(const ExecutionSpace&, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_label, arg_alloc_size, arg_logical_size);
  }
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return impl_allocate(arg_label, arg_alloc_size, arg_logical_size,
                         Kokkos::Tools::make_space_handle(Space::name()));
  }

  void deallocate(void* const arg_alloc_ptr,
                  const size_t arg_alloc_size) const {
    deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
  }
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const {
    impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size,
                    Kokkos::Tools::make_space_handle(Space::name()));
  }

 private:
  static void* impl_allocate(const char* arg_label, size_t arg_alloc_size,
                             size_t arg_logical_size,
                             Kokkos::Tools::SpaceHandle arg_handle);
  static void impl_deallocate(const char* arg_label, void* arg_alloc_ptr,
                              size_t arg_alloc_size, size_t arg_logical_size,
                              Kokkos::Tools::SpaceHandle arg_handle);
};
}  // namespace Impl

class MetalSpace : public Impl::MetalSpaceBase<MetalSpace> {
 public:
  static constexpr const char* name() { return "MetalSpace"; }
};

class MetalSharedSpace : public Impl::MetalSpaceBase<MetalSharedSpace> {
 public:
  static constexpr const char* name() { return "MetalSharedSpace"; }
};

}  // namespace Kokkos::Experimental

namespace Kokkos::Impl {

template <class T>
struct is_metal_type_space
    : std::bool_constant<
          std::is_same_v<T, Kokkos::Experimental::MetalSpace> ||
          std::is_same_v<T, Kokkos::Experimental::MetalSharedSpace>> {};

template <>
struct MemorySpaceAccess<HostSpace, Kokkos::Experimental::MetalSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
};

template <>
struct MemorySpaceAccess<HostSpace, Kokkos::Experimental::MetalSharedSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MetalSpace, HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MetalSharedSpace, HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MetalSpace,
                         Kokkos::Experimental::MetalSharedSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MetalSharedSpace,
                         Kokkos::Experimental::MetalSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
};

}  // namespace Kokkos::Impl

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::Experimental::MetalSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(
    Kokkos::Experimental::MetalSharedSpace);

#endif
