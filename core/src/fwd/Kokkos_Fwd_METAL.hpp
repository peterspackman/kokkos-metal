// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_FWD_HPP_
#define KOKKOS_METAL_FWD_HPP_

#if defined(KOKKOS_ENABLE_METAL)
namespace Kokkos::Experimental {
class Metal;             ///< Execution space on Apple GPUs
class MetalSpace;        ///< GPU memory
class MetalSharedSpace;  ///< GPU memory the host can access too
}  // namespace Kokkos::Experimental
#endif
#endif
