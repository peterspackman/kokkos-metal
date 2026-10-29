// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_MDRANGEPOLICY_HPP
#define KOKKOS_METAL_MDRANGEPOLICY_HPP

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {

// Consecutive threads take consecutive first indices, as LayoutLeft wants.
template <>
struct default_outer_direction<Kokkos::Experimental::Metal> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<Kokkos::Experimental::Metal> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace Impl {

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Kokkos::Experimental::Metal,
                                ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // namespace Impl
}  // namespace Kokkos

#endif
