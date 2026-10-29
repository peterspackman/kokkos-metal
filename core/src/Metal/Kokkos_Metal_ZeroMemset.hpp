// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_ZEROMEMSET_HPP
#define KOKKOS_METAL_ZEROMEMSET_HPP

#include <Metal/Kokkos_Metal.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos::Impl {

template <>
struct ZeroMemset<Kokkos::Experimental::Metal> {
  ZeroMemset(const Kokkos::Experimental::Metal&, void* dst, size_t cnt) {
    MetalRuntime::fill_zero(dst, cnt);
  }
};

}  // namespace Kokkos::Impl

#endif
