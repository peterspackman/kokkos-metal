// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_UNIQUE_TOKEN_HPP
#define KOKKOS_METAL_UNIQUE_TOKEN_HPP

#include <Kokkos_UniqueToken.hpp>
#include <Kokkos_View.hpp>
#include <Metal/Kokkos_Metal.hpp>

namespace Kokkos {

namespace Impl {
Kokkos::View<uint32_t*, Kokkos::Experimental::MetalSpace>
metal_global_unique_token_locks(bool deallocate = false);
}

namespace Experimental {

// A token is a lock word claimed by compare-and-swap, starting from the
// thread's own index. A thread that loses moves on to the next word; one that
// finds none free aborts rather than spin, which could hang the GPU.
template <>
class UniqueToken<Metal, UniqueTokenScope::Global> {
 protected:
  View<uint32_t*, MetalSpace> m_locks;

 public:
  using execution_space = Metal;
  using size_type       = int32_t;

  explicit UniqueToken(execution_space const& = Metal())
      : m_locks(Kokkos::Impl::metal_global_unique_token_locks()) {}

 protected:
  explicit UniqueToken(size_type max_size)
      : m_locks("Kokkos::UniqueToken::m_locks", max_size) {}
  UniqueToken(size_type max_size, execution_space const& exec)
      : m_locks(Kokkos::view_alloc(exec, "Kokkos::UniqueToken::m_locks"),
                max_size) {}

 public:
  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept { return m_locks.extent(0); }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  size_type acquire() const {
    size_type idx = 0;
    KOKKOS_IF_ON_DEVICE((
        const size_type n = size(); idx = Metal::impl_hardware_thread_id() % n;
        for (size_type tries = 0;
             Kokkos::atomic_compare_exchange(m_locks.data() + idx, 0u, 1u) !=
             0u;
             idx = (idx + 33) % n) {
          // The lanes holding tokens may be waiting on this one to leave the
          // loop (a SIMD group runs in lockstep): never wait for a token.
          if (++tries > 2 * n)
            Kokkos::abort("Kokkos::UniqueToken<Metal>: no free token");
        }
        desul::atomic_thread_fence(desul::MemoryOrderAcquire(),
                                   desul::MemoryScopeDevice());))
    return idx;
  }

  /// \brief release an acquired value
  KOKKOS_INLINE_FUNCTION
  void release(size_type idx) const noexcept {
    desul::atomic_thread_fence(desul::MemoryOrderRelease(),
                               desul::MemoryScopeDevice());
    (void)Kokkos::atomic_exchange(m_locks.data() + idx, 0u);
  }
};

template <>
class UniqueToken<Metal, UniqueTokenScope::Instance>
    : public UniqueToken<Metal, UniqueTokenScope::Global> {
 public:
  UniqueToken()
      : UniqueToken<Metal, UniqueTokenScope::Global>(Metal().concurrency()) {}
  explicit UniqueToken(execution_space const& arg)
      : UniqueToken<Metal, UniqueTokenScope::Global>(arg.concurrency(), arg) {}
  explicit UniqueToken(size_type max_size)
      : UniqueToken<Metal, UniqueTokenScope::Global>(max_size) {}
  UniqueToken(size_type max_size, execution_space const& arg)
      : UniqueToken<Metal, UniqueTokenScope::Global>(max_size, arg) {}
};

}  // namespace Experimental
}  // namespace Kokkos

#endif
