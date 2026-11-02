// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Translation of one fully inlined kernel entry from LLVM IR to Metal
// Shading Language.

#pragma once

#include <stdexcept>
#include <string>

namespace llvm {
class Function;
}

namespace kmetal {

// Thrown for IR the translation does not support; the message names the
// construct, and the plugin reports it as a compile error.
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct KernelSource {
  std::string source;  // program-scope constants and the kernel function
  bool uses_threadgroup_memory = false;
  bool uses_abort              = false;  // binds the abort flag as buffer 1
  bool uses_printf             = false;  // binds the printf buffer as buffer 2
};

// `F` takes one pointer argument (the driver, bound as buffer 0) and calls
// nothing but __kmetal_* functions and LLVM intrinsics.
KernelSource emit_kernel(llvm::Function& F, const std::string& name);

// Helpers every kernel's source starts with.
const std::string& prelude();

}  // namespace kmetal
