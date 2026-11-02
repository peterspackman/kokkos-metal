// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_WORKGRAPHPOLICY_HPP
#define KOKKOS_METAL_WORKGRAPHPOLICY_HPP

#include <Metal/Kokkos_Metal_Parallel_Range.hpp>

namespace Kokkos::Impl {

// Every thread pops ready work until the graph completes. pop_work never
// waits, so lanes that find no work just go round again with the others.
template <class Policy, class F>
struct MetalWorkGraphDriver {
  F functor;
  Policy policy;

  void metal_run() const {
    // A graph that never completes (a cycle) would spin forever and wedge the
    // GPU: give up after a stretch without work.
    uint32_t idle = 0;
    for (std::int32_t w;
         Policy::COMPLETED_TOKEN != (w = policy.pop_work());) {
      if (w == Policy::END_TOKEN) {
        if (++idle == (1u << 20))
          Kokkos::abort("Kokkos::Metal: WorkGraphPolicy made no progress");
        continue;
      }
      idle = 0;
      metal_invoke<typename Policy::work_tag>(functor, w);
      policy.completed_work(w);
    }
  }
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::WorkGraphPolicy<Traits...>,
                  Kokkos::Experimental::Metal> {
  using Policy = Kokkos::WorkGraphPolicy<Traits...>;
  Policy m_policy;
  FunctorType m_functor;

 public:
  ParallelFor(const FunctorType& functor, const Policy& policy)
      : m_policy(policy), m_functor(functor) {}

  void execute() const {
    using Driver           = MetalWorkGraphDriver<Policy, FunctorType>;
    const uint32_t threads = metal_threads<Driver>(256);
    metal_launch(Driver{m_functor, m_policy}, 64, threads);
  }
};

}  // namespace Kokkos::Impl

#endif
