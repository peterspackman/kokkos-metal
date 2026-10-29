// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DECLARE_METAL_HPP
#define KOKKOS_DECLARE_METAL_HPP

#if defined(KOKKOS_ENABLE_METAL)
#include <Metal/Kokkos_Metal.hpp>
#include <Metal/Kokkos_MetalSpace.hpp>
#include <Metal/Kokkos_Metal_DeepCopy.hpp>
#include <Metal/Kokkos_Metal_ZeroMemset.hpp>
#include <Metal/Kokkos_Metal_MDRangePolicy.hpp>
#include <Metal/Kokkos_Metal_Parallel_Range.hpp>
#include <Metal/Kokkos_Metal_Parallel_MDRange.hpp>
#include <Metal/Kokkos_Metal_Parallel_Team.hpp>
#endif

#endif
