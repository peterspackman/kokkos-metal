# Kokkos on Apple GPUs (Metal)

`Kokkos::Experimental::Metal`: an execution space for the GPU of an
Apple-silicon Mac. Ordinary Kokkos code -- lambdas, functors, Views,
RangePolicy, MDRangePolicy, TeamPolicy with scratch and nested parallelism --
compiles and runs on it unchanged.

## Building

You need Homebrew's LLVM (clang 21 here) and macOS 15 or later; Xcode's Metal
toolchain is not needed.

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_METAL=ON
```

Metal is then the default execution space, and `SharedSpace` and
`SharedHostPinnedSpace` are `MetalSharedSpace`. The tests:

```bash
cmake -S metal -B build-metal -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build-metal && ctest --test-dir build-metal
```

## How it works

- **A compiler plugin, not a second compiler.** Every translation unit using
  Kokkos is compiled once, by clang, with `-fpass-plugin=libKokkosMetalPlugin`
  (Kokkos adds the flag to its targets; the plugin is built when Kokkos
  configures). At the start of the optimisation pipeline the plugin clones
  each kernel the backend instantiated, with everything it calls, into a
  device module, optimises it there, and translates it to Metal Shading
  Language (`core/src/Metal/plugin`). The host object keeps the MSL and a
  static constructor registering it; the OS compiles each kernel on its first
  launch.
- **One address for host and GPU.** Every allocation is a shared MTLBuffer whose
  pages are also mapped on the host at the buffer's GPU address, so a pointer
  is valid on both sides and functors are copied to the GPU bitwise, as on
  CUDA. A residency set keeps the buffers resident.
- **The translation.** Control flow is rebuilt from the dominator tree (MSL has
  no `goto`; Ramsey, "Beyond Relooper"), address spaces are inferred over the
  IR, and every device function and call is marked convergent so that the
  optimiser cannot make a barrier or a shuffle conditional on a lane. Maths
  calls Metal's `precise::` functions: the default ones are fast-math (`tanh(50)`
  is NaN).
- **Teams.** A team is a threadgroup of `team_size * vector_length` threads, a
  thread's vector lanes within one SIMD group; level-0 scratch is threadgroup
  memory, level 1 device memory.
- **GEMMs.** `Kokkos::Experimental::metal_gemm` is the vendor GEMM, as cuBLAS is
  for CUDA: MPSGraph for large shapes (with the bias in the same kernel),
  MPSMatrixMultiplication for small or strided ones.

## Limitations

- Apple GPUs have no fp64. Doubles in a kernel are emulated in software,
  exactly (IEEE, round to nearest even) but slowly; transcendentals go through
  float. Compute in float where it matters.
- Metal has no 64-bit atomics but unsigned min and max: atomics on doubles and
  64-bit integers do not run. Kernels the plugin cannot translate (these,
  function pointers, virtual calls, calls into other translation units) warn
  when compiled and abort if launched.
- One command queue: every instance of the execution space shares it.
- Kokkos's own unit tests are not ported; `metal/` has the backend's tests.

## Debugging

| Variable | |
|---|---|
| `KOKKOS_METAL_DUMP=dir` | write each kernel's MSL to `dir` as it is compiled |
| `KOKKOS_METAL_PLUGIN_DUMP=1` | (at compile time) print the MSL of every kernel |
| `KOKKOS_METAL_STATS=1` | per-kernel GPU times at exit (serialises the kernels) |
| `KOKKOS_METAL_BUSY=1` | GPU-busy against wall time at exit |
| `KOKKOS_METAL_NANCHECK=1` | report kernels that write NaNs (slow) |
| `KOKKOS_METAL_GEMM=mps\|graph` | one GEMM path for every shape |
