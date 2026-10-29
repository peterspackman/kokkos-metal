// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_PARALLEL_MDRANGE_HPP
#define KOKKOS_METAL_PARALLEL_MDRANGE_HPP

#include <KokkosExp_MDRangePolicy.hpp>
#include <Metal/Kokkos_Metal_Parallel_Range.hpp>

#include <utility>

// MDRangePolicy as a flat index space: the flat index is split into one per
// rank, the inner direction's fastest. Tiles are ignored; consecutive threads
// take consecutive inner indices.

namespace Kokkos::Impl {

template <class Policy>
struct MetalMDShape {
  static constexpr int rank = Policy::rank;
  using index_type          = typename Policy::index_type;

  index_type lower[rank];
  index_type extent[rank];
  uint64_t count;

  explicit MetalMDShape(const Policy& policy) : count(1) {
    for (int d = 0; d < rank; ++d) {
      lower[d]  = policy.m_lower[d];
      extent[d] = policy.m_upper[d] > policy.m_lower[d]
                      ? policy.m_upper[d] - policy.m_lower[d]
                      : 0;
      count *= uint64_t(extent[d]);
    }
  }

  uint64_t size() const { return count; }

  // K is the counter type: with 32 bits, so are the divisions.
  template <class Tag, class F, class K, class... Args>
  void call(const F& f, K k, Args&&... args) const {
    index_type idx[rank];
    if constexpr (Policy::inner_direction == Iterate::Left) {
      for (int d = 0; d < rank; ++d) {
        idx[d] = lower[d] + index_type(k % K(extent[d]));
        k /= K(extent[d]);
      }
    } else {
      for (int d = rank - 1; d >= 0; --d) {
        idx[d] = lower[d] + index_type(k % K(extent[d]));
        k /= K(extent[d]);
      }
    }
    expand<Tag>(f, idx, std::make_index_sequence<rank>{},
                static_cast<Args&&>(args)...);
  }

 private:
  template <class Tag, class F, size_t... I, class... Args>
  static void expand(const F& f, const index_type* idx,
                     std::index_sequence<I...>, Args&&... args) {
    metal_invoke<Tag>(f, idx[I]..., static_cast<Args&&>(args)...);
  }
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::Metal> {
  using Policy = Kokkos::MDRangePolicy<Traits...>;
  FunctorType m_functor;
  Policy m_policy;

 public:
  ParallelFor(const FunctorType& functor, const Policy& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    metal_for<typename Policy::work_tag>(MetalMDShape<Policy>(m_policy),
                                         m_functor,
                                         metal_launch_bound<Policy>());
  }
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     Kokkos::MDRangePolicy<Traits...>,
                     Kokkos::Experimental::Metal> {
  using Policy = Kokkos::MDRangePolicy<Traits...>;
  using pointer_type =
      typename CombinedFunctorReducerType::reducer_type::pointer_type;
  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  pointer_type m_result;
  bool m_result_on_device;

 public:
  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& functor_reducer,
                 const Policy& policy, const ViewType& result)
      : m_functor_reducer(functor_reducer),
        m_policy(policy),
        m_result(result.data()),
        m_result_on_device(
            MemorySpaceAccess<Kokkos::Experimental::MetalSpace,
                              typename ViewType::memory_space>::accessible) {}

  void execute() const {
    metal_reduce<typename Policy::work_tag>(
        MetalMDShape<Policy>(m_policy), m_functor_reducer, m_result,
        m_result_on_device, metal_launch_bound<Policy>());
  }
};

}  // namespace Kokkos::Impl

#endif
