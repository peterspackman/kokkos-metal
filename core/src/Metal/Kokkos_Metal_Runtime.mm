// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <Metal/Kokkos_Metal_Runtime.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if !__has_feature(objc_arc)
#error "Kokkos_Metal_Runtime.mm must be compiled with -fobjc-arc"
#endif

namespace Kokkos::Impl::MetalRuntime {
namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "Kokkos::Metal: %s\n", msg.c_str());
  std::fflush(stderr);
  std::abort();
}

// Kernels registered before initialize() (static constructors) live here.
struct Registry {
  std::mutex mutex;
  std::unordered_map<const void*, KernelDesc> kernels;
};
Registry& registry() {
  static Registry r;
  return r;
}

struct Kernel {
  KernelDesc desc;
  id<MTLComputePipelineState> pipeline = nil;
};

// Copy and fill on the GPU, by address: every allocation is mapped at the
// same address on both sides, so no buffer lookup is needed.
constexpr const char* kBuiltinSource = R"(
#include <metal_stdlib>
using namespace metal;
struct CopyArgs { ulong dst; ulong src; ulong n; };
kernel void kmetal_copy16(constant CopyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n / 16) ((device uint4*)a.dst)[i] = ((device const uint4*)a.src)[i];
}
kernel void kmetal_copy1(constant CopyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n) ((device uchar*)a.dst)[i] = ((device const uchar*)a.src)[i];
}
kernel void kmetal_zero16(constant CopyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n / 16) ((device uint4*)a.dst)[i] = uint4(0);
}
kernel void kmetal_zero1(constant CopyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n) ((device uchar*)a.dst)[i] = 0;
}
// A GEMM's bias: C[i, j] = bias[j] (before), or += (after).
struct BiasArgs { ulong c, bias; uint m, n, ldc, add; };
kernel void kmetal_bias(constant BiasArgs& a [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.n || p.y >= a.m) return;
  device float* c = (device float*)a.c + p.y * a.ldc + p.x;
  const float b = ((device const float*)a.bias)[p.x];
  *c = a.add ? *c + b : b;
}
)";

// Dispatches per command buffer before it is committed, so the GPU starts on
// work while the host is still enqueueing.
constexpr int kDispatchesPerCommandBuffer = 32;

struct State {
  id<MTLDevice> device                 = nil;
  id<MTLCommandQueue> queue            = nil;
  id<MTLResidencySet> residency        = nil;
  id<MTLCommandBuffer> command_buffer  = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  id<MTLCommandBuffer> last_committed  = nil;
  // Set by a kernel that called Kokkos::abort; checked at each fence.
  id<MTLBuffer> abort_flag = nil;
  int dispatches           = 0;
  bool residency_dirty     = false;

  std::recursive_mutex mutex;
  std::unordered_map<const void*, Kernel> kernels;
  // Ordered, to find the buffer holding an address (for MPS).
  std::map<void*, id<MTLBuffer>> allocations;
  // Command buffers wait on their predecessor: resources are untracked, so
  // nothing else orders them.
  id<MTLEvent> chain   = nil;
  uint64_t chain_value = 0;
  std::map<std::tuple<bool, bool, int, int, int, float, float, int, int, int>,
           MPSMatrixMultiplication*>
      gemms;
  // MPSGraph GEMMs (op(A) op(B) * alpha + bias), compiled per shape.
  struct GraphGemm {
    MPSGraphExecutable* executable;
    MPSGraphTensor *a, *b, *bias;
  };
  std::map<std::tuple<bool, bool, int, int, int, float, bool>, GraphGemm> graph_gemms;
  MPSGraphDevice* graph_device = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> builtins;
  MTLCompileOptions* options = nil;
  const char* dump_dir       = nullptr;
  // KOKKOS_METAL_STATS: counts and times, printed at finalize.
  bool stats = false;
  // KOKKOS_METAL_NANCHECK: after each kernel, report allocations that newly
  // hold a float NaN (a debugging aid: slow).
  bool nancheck = false;
  // KOKKOS_METAL_BUSY: how long the GPU was busy, from each command buffer's
  // own times, without serialising anything.
  bool busy = false;
  std::atomic<double> t_busy{0};
  double t_start = 0;
  std::unordered_map<void*, bool> had_nan;
  uint64_t n_dispatch = 0, n_fence = 0, n_compile = 0, n_alloc = 0, n_free = 0, n_busy_free = 0;
  double t_compile = 0, t_fence = 0;
  std::unordered_map<std::string, std::pair<uint64_t, double>> per_kernel;  // count, GPU s
};
State* g_state = nullptr;

State& state() {
  if (!g_state) fail("used before Kokkos::initialize");
  return *g_state;
}

MTLCompileOptions* make_compile_options() {
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion    = MTLLanguageVersion3_2;
  // Kokkos code expects IEEE results, not fast-math ones.
  o.mathMode = getenv("KOKKOS_METAL_FAST_MATH") ? MTLMathModeFast : MTLMathModeSafe;
  return o;
}

std::string numbered(const std::string& src) {
  std::string out;
  int line     = 1;
  size_t start = 0;
  while (start < src.size()) {
    size_t end = src.find('\n', start);
    if (end == std::string::npos) end = src.size();
    char buf[16];
    std::snprintf(buf, sizeof buf, "%5d  ", line++);
    out += buf;
    out.append(src, start, end - start);
    out += '\n';
    start = end + 1;
  }
  return out;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

id<MTLComputePipelineState> compile(State& s, const std::string& source, const char* name) {
  const double t0 = now();
  struct Timer {
    State& s;
    double t0;
    ~Timer() { s.t_compile += now() - t0, ++s.n_compile; }
  } timer{s, t0};
  if (s.dump_dir) {
    std::string path = std::string(s.dump_dir) + "/" + name + ".metal";
    if (FILE* f = std::fopen(path.c_str(), "w")) {
      std::fwrite(source.data(), 1, source.size(), f);
      std::fclose(f);
    }
  }
  NSError* error     = nil;
  id<MTLLibrary> lib = [s.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                              options:s.options
                                                error:&error];
  if (!lib)
    fail(std::string("compiling kernel ") + name + " failed:\n" +
         error.localizedDescription.UTF8String + "\n--- source ---\n" + numbered(source));
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) fail(std::string("kernel ") + name + " missing from its library");
  id<MTLComputePipelineState> pso = [s.device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso)
    fail(std::string("pipeline for ") + name + " failed: " + error.localizedDescription.UTF8String);
  return pso;
}

Kernel& kernel(State& s, const void* key) {
  auto it = s.kernels.find(key);
  if (it == s.kernels.end()) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    auto d = r.kernels.find(key);
    if (d == r.kernels.end())
      fail("launch of an unregistered kernel: was this translation unit compiled "
           "with the kokkos-metal plugin (-fpass-plugin)?");
    it = s.kernels.emplace(key, Kernel{d->second}).first;
  }
  Kernel& k = it->second;
  if (k.desc.flags & kUnsupported) fail(k.desc.source);
  if (!k.pipeline)
    k.pipeline = compile(s, std::string(k.desc.prelude) + k.desc.source, k.desc.name);
  return k;
}

id<MTLComputePipelineState> builtin(State& s, const char* name) {
  auto it = s.builtins.find(name);
  if (it != s.builtins.end()) return it->second;
  return s.builtins[name] = compile(s, kBuiltinSource, name);
}

void begin(State& s) {
  if (s.command_buffer) return;
  s.command_buffer = [s.queue commandBufferWithUnretainedReferences];
  if (s.chain_value) [s.command_buffer encodeWaitForEvent:s.chain value:s.chain_value];
  s.dispatches = 0;
}

id<MTLComputeCommandEncoder> encoder(State& s) {
  if (s.residency_dirty) {
    [s.residency commit];
    s.residency_dirty = false;
  }
  begin(s);
  if (!s.encoder)
    s.encoder = [s.command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  return s.encoder;
}

void check(id<MTLCommandBuffer> cb) {
  if (cb.status == MTLCommandBufferStatusError)
    fail(std::string("GPU command buffer failed: ") + cb.error.localizedDescription.UTF8String);
}

void commit(State& s) {
  if (!s.command_buffer) return;
  if (s.encoder) [s.encoder endEncoding];
  [s.command_buffer encodeSignalEvent:s.chain value:++s.chain_value];
  if (s.busy) {
    std::atomic<double>* total = &s.t_busy;
    [s.command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      double t = cb.GPUEndTime - cb.GPUStartTime, old = total->load();
      while (!total->compare_exchange_weak(old, old + t)) {
      }
    }];
  }
  [s.command_buffer commit];
  s.last_committed = s.command_buffer;
  s.command_buffer = nil;
  s.encoder        = nil;
}

void after_dispatch(State& s) {
  ++s.n_dispatch;
  if (++s.dispatches >= kDispatchesPerCommandBuffer) commit(s);
}

void builtin_dispatch(State& s, const char* name, uint64_t dst, uint64_t src, uint64_t n,
                      uint64_t threads) {
  if (threads == 0) return;
  id<MTLComputePipelineState> pso  = builtin(s, name);
  id<MTLComputeCommandEncoder> enc = encoder(s);
  uint64_t args[3]                 = {dst, src, n};
  [enc setComputePipelineState:pso];
  [enc setBytes:args length:sizeof args atIndex:0];
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  after_dispatch(s);
}

}  // namespace

void initialize() {
  if (g_state) return;
  @autoreleasepool {
    auto* s   = new State;
    s->device = MTLCreateSystemDefaultDevice();
    if (!s->device) fail("no Metal device");
    if (!s->device.hasUnifiedMemory) fail("the Metal backend needs unified memory");
    s->queue                      = [s->device newCommandQueue];
    s->chain                      = [s->device newEvent];
    MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
    rd.label                      = @"Kokkos::Metal";
    NSError* error                = nil;
    s->residency                  = [s->device newResidencySetWithDescriptor:rd error:&error];
    if (!s->residency) fail("cannot create a residency set");
    [s->queue addResidencySet:s->residency];
    s->options    = make_compile_options();
    s->abort_flag = [s->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    s->dump_dir   = getenv("KOKKOS_METAL_DUMP");
    s->stats      = getenv("KOKKOS_METAL_STATS") != nullptr;
    s->nancheck   = getenv("KOKKOS_METAL_NANCHECK") != nullptr;
    s->busy       = getenv("KOKKOS_METAL_BUSY") != nullptr;
    s->t_start    = now();
    g_state       = s;
  }
}

void finalize() {
  if (!g_state) return;
  fence();
  if (g_state->busy)
    std::fprintf(stderr, "Kokkos::Metal: GPU busy %.3f s of %.3f s, %llu fences\n",
                 g_state->t_busy.load(), now() - g_state->t_start,
                 (unsigned long long)g_state->n_fence);
  if (g_state->stats)
    std::fprintf(stderr,
                 "Kokkos::Metal stats: %llu dispatches, %llu fences (%.3f s waiting), %llu kernels "
                 "compiled (%.3f s), %llu allocations, %llu frees (%llu with work in flight)\n",
                 (unsigned long long)g_state->n_dispatch, (unsigned long long)g_state->n_fence,
                 g_state->t_fence, (unsigned long long)g_state->n_compile, g_state->t_compile,
                 (unsigned long long)g_state->n_alloc, (unsigned long long)g_state->n_free,
                 (unsigned long long)g_state->n_busy_free);
  if (g_state->stats) {
    std::vector<std::pair<double, std::string>> rows;
    for (auto& [label, ct] : g_state->per_kernel) {
      char buf[64];
      std::snprintf(buf, sizeof buf, "%6llu x %9.3f ms  ", (unsigned long long)ct.first,
                    ct.second * 1e3);
      rows.push_back({ct.second, buf + label});
    }
    std::sort(rows.rbegin(), rows.rend());
    for (size_t i = 0; i < rows.size() && i < 25; ++i)
      std::fprintf(stderr, "  %s\n", rows[i].second.c_str());
  }
  {
    std::lock_guard<std::recursive_mutex> lock(g_state->mutex);
    for (auto& [ptr, buf] : g_state->allocations)
      mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, buf.length);
    g_state->allocations.clear();
  }
  delete g_state;
  g_state = nullptr;
}

bool is_initialized() { return g_state != nullptr; }

const char* device_name() { return state().device.name.UTF8String; }

void* allocate(size_t bytes) {
  if (bytes == 0) return nullptr;
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  // Whole pages: Metal packs smaller buffers into shared pages, which could
  // not each be mapped at their own GPU address.
  bytes             = (bytes + vm_page_size - 1) / vm_page_size * vm_page_size;
  id<MTLBuffer> buf = [s.device
      newBufferWithLength:bytes
                  options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
  if (!buf) return nullptr;
  // Map the buffer's pages on the host at its GPU address too, so one
  // pointer is valid on both sides.
  mach_vm_address_t addr = buf.gpuAddress;
  vm_prot_t cur, max;
  kern_return_t kr =
      mach_vm_remap(mach_task_self(), &addr, buf.length, 0, VM_FLAGS_FIXED, mach_task_self(),
                    (mach_vm_address_t)buf.contents, false, &cur, &max, VM_INHERIT_NONE);
  if (kr != KERN_SUCCESS || addr != buf.gpuAddress)
    fail("cannot map a Metal buffer at its GPU address (mach_vm_remap: " + std::to_string(kr) +
         ")");
  void* ptr = reinterpret_cast<void*>(addr);
  ++s.n_alloc;
  s.allocations[ptr] = buf;
  [s.residency addAllocation:buf];
  s.residency_dirty = true;
  return ptr;
}

void deallocate(void* ptr) {
  if (!ptr || !g_state) return;
  State& s = state();
  if (s.command_buffer || s.last_committed) ++s.n_busy_free;
  fence();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  auto it = s.allocations.find(ptr);
  if (it == s.allocations.end()) fail("deallocate of a pointer Metal did not allocate");
  ++s.n_free;
  mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, it->second.length);
  [s.residency removeAllocation:it->second];
  s.residency_dirty = true;
  s.allocations.erase(it);
}

void dispatch(const void* key, const void* args, size_t args_size, const uint32_t groups[3],
              const uint32_t threads[3], size_t threadgroup_bytes) {
  if (uint64_t(groups[0]) * groups[1] * groups[2] == 0) return;
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  @autoreleasepool {
    Kernel& k                        = kernel(s, key);
    id<MTLComputeCommandEncoder> enc = encoder(s);
    [enc setComputePipelineState:k.pipeline];
    if (args_size <= 4096) {
      [enc setBytes:args length:args_size atIndex:0];
    } else {
      id<MTLBuffer> b = [s.device newBufferWithBytes:args
                                              length:args_size
                                             options:MTLResourceStorageModeShared];
      [enc setBuffer:b offset:0 atIndex:0];
      // The command buffer does not retain its resources; this one must
      // outlive it.
      [s.command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        (void)b;
      }];
    }
    if (k.desc.flags & kUsesAbort) [enc setBuffer:s.abort_flag offset:0 atIndex:1];
    if (k.desc.flags & kUsesThreadgroupMemory)
      [enc setThreadgroupMemoryLength:(threadgroup_bytes + 15) / 16 * 16 ?: 16 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
        threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
    after_dispatch(s);
    if (s.nancheck) {
      id<MTLCommandBuffer> cb = s.command_buffer;
      commit(s);
      [cb waitUntilCompleted];
      for (auto& [ptr, buf] : s.allocations) {
        const uint32_t* w = static_cast<const uint32_t*>(ptr);
        size_t n = buf.length / 4, first = n;
        for (size_t i = 0; i < n; ++i)
          if ((w[i] & 0x7fffffffu) > 0x7f800000u) {
            first = i;
            break;
          }
        bool nan = first < n, &had = s.had_nan[ptr];
        if (nan && !had)
          std::fprintf(stderr, "NaN in %p (+%zu of %zu words) after %.200s\n", ptr, first, n,
                       k.desc.source);
        had = nan;
      }
    }
    if (s.stats) {
      // Alone in its command buffer, timed on the GPU.
      id<MTLCommandBuffer> cb = s.command_buffer;
      commit(s);
      [cb waitUntilCompleted];
      std::string label(k.desc.source, strcspn(k.desc.source, "\n"));
      auto& [count, time] = s.per_kernel[label.substr(0, 160)];
      ++count;
      time += cb.GPUEndTime - cb.GPUStartTime;
    }
  }
}

uint32_t max_threads_per_threadgroup(const void* key) {
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  return kernel(s, key).pipeline.maxTotalThreadsPerThreadgroup;
}

uint32_t simd_width(const void* key) {
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  return kernel(s, key).pipeline.threadExecutionWidth;
}

size_t max_threadgroup_memory() { return state().device.maxThreadgroupMemoryLength; }

void copy(void* dst, const void* src, size_t bytes) {
  if (bytes == 0 || dst == src) return;
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  @autoreleasepool {
    auto d = uint64_t(dst), r = uint64_t(src);
    if ((d | r | bytes) % 16 == 0)
      builtin_dispatch(s, "kmetal_copy16", d, r, bytes, bytes / 16);
    else
      builtin_dispatch(s, "kmetal_copy1", d, r, bytes, bytes);
  }
}

void fill_zero(void* dst, size_t bytes) {
  if (bytes == 0) return;
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  @autoreleasepool {
    auto d = uint64_t(dst);
    if ((d | bytes) % 16 == 0)
      builtin_dispatch(s, "kmetal_zero16", d, 0, bytes, bytes / 16);
    else
      builtin_dispatch(s, "kmetal_zero1", d, 0, bytes, bytes);
  }
}

// The buffer holding [ptr, ptr + bytes) and the offset of ptr in it.
std::pair<id<MTLBuffer>, size_t> find_buffer(State& s, const void* ptr, size_t bytes) {
  auto it = s.allocations.upper_bound(const_cast<void*>(ptr));
  if (it != s.allocations.begin()) {
    --it;
    size_t off = static_cast<const char*>(ptr) - static_cast<const char*>(it->first);
    if (off + bytes <= it->second.length) return {it->second, off};
  }
  fail("a GEMM operand is not in Metal memory");
}

void bias_pass(State& s, float* c, const float* bias, int m, int n, int ldc, bool add) {
  struct {
    uint64_t c, bias;
    uint32_t m, n, ldc, add;
  } args{uint64_t(c), uint64_t(bias), uint32_t(m), uint32_t(n), uint32_t(ldc), add};
  id<MTLComputeCommandEncoder> enc = encoder(s);
  [enc setComputePipelineState:builtin(s, "kmetal_bias")];
  [enc setBytes:&args length:sizeof args atIndex:0];
  [enc dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
  after_dispatch(s);
}

// Orders what is encoded next in the command buffer after what was before:
// with untracked resources, encoders are otherwise free to overlap.
void encoder_barrier(State& s) {
  if (s.encoder) {
    [s.encoder endEncoding];
    s.encoder = nil;
  }
  ++s.chain_value;
  [s.command_buffer encodeSignalEvent:s.chain value:s.chain_value];
  [s.command_buffer encodeWaitForEvent:s.chain value:s.chain_value];
}

MPSMatrix* mps_matrix(State& s, const float* p, int rows, int cols, int ld) {
  auto [buf, off]        = find_buffer(s, p, (size_t(rows - 1) * ld + cols) * sizeof(float));
  MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                 columns:cols
                                                                rowBytes:ld * sizeof(float)
                                                                dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:buf offset:off descriptor:d];
}

// A row-major matrix (or, with cols 0, a vector) for MPSGraph: an MPSNDArray,
// since tensor data made from an MPSMatrix ignores the matrix's offset in its
// buffer, and a View's data never starts its allocation.
MPSGraphTensorData* graph_data(State& s, const float* p, int rows, int cols) {
  const size_t count      = cols ? size_t(rows) * cols : size_t(rows);
  auto [buf, off]         = find_buffer(s, p, count * sizeof(float));
  NSArray* shape          = cols ? @[ @(rows), @(cols) ] : @[ @(rows) ];
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                   shape:shape];
  d.preferPackedRows      = YES;  // else rows are padded to an alignment
  MPSNDArray* nd          = [[MPSNDArray alloc] initWithBuffer:buf offset:off descriptor:d];
  return [[MPSGraphTensorData alloc] initWithMPSNDArray:nd];
}

// C = alpha op(A) op(B) + bias by MPSGraph, which is faster than
// MPSMatrixMultiplication and adds the bias in the same kernel. MPSGraph
// commits the command buffer it is given whenever it likes, so it gets one of
// its own, chained by the event after the kernels before; the event's next
// value, signalled at the end of what it leaves, orders the kernels after.
void gemm_graph(State& s, bool ta, bool tb, int m, int n, int k, float alpha, const float* a,
                int lda, const float* b, int ldb, float* c, int ldc, const float* bias) {
  State::GraphGemm& gg = s.graph_gemms[{ta, tb, m, n, k, alpha, bias != nullptr}];
  if (!gg.executable) {
    if (!s.graph_device) s.graph_device = [MPSGraphDevice deviceWithMTLDevice:s.device];
    MPSGraph* g         = [MPSGraph new];
    auto shape          = [](int r, int c) { return @[ @(r), @(c) ]; };
    NSArray* sa         = ta ? shape(k, m) : shape(m, k);
    NSArray* sb         = tb ? shape(n, k) : shape(k, n);
    gg.a                = [g placeholderWithShape:sa dataType:MPSDataTypeFloat32 name:nil];
    gg.b                = [g placeholderWithShape:sb dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* lhs = ta ? [g transposeTensor:gg.a dimension:0 withDimension:1 name:nil] : gg.a;
    MPSGraphTensor* rhs = tb ? [g transposeTensor:gg.b dimension:0 withDimension:1 name:nil] : gg.b;
    MPSGraphTensor* out = [g matrixMultiplicationWithPrimaryTensor:lhs
                                                   secondaryTensor:rhs
                                                              name:nil];
    if (alpha != 1.0f)
      out = [g multiplicationWithPrimaryTensor:out
                               secondaryTensor:[g constantWithScalar:alpha
                                                            dataType:MPSDataTypeFloat32]
                                          name:nil];
    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    feeds[gg.a] = [[MPSGraphShapedType alloc] initWithShape:sa dataType:MPSDataTypeFloat32];
    feeds[gg.b] = [[MPSGraphShapedType alloc] initWithShape:sb dataType:MPSDataTypeFloat32];
    if (bias) {
      gg.bias        = [g placeholderWithShape:@[ @(n) ] dataType:MPSDataTypeFloat32 name:nil];
      out            = [g additionWithPrimaryTensor:out secondaryTensor:gg.bias name:nil];
      feeds[gg.bias] = [[MPSGraphShapedType alloc] initWithShape:@[ @(n) ]
                                                        dataType:MPSDataTypeFloat32];
    }
    gg.executable = [g compileWithDevice:s.graph_device
                                   feeds:feeds
                           targetTensors:@[ out ]
                        targetOperations:nil
                   compilationDescriptor:nil];
  }
  // The inputs in the executable's order of its feeds.
  NSMutableArray* inputs = [NSMutableArray array];
  for (MPSGraphTensor* t in gg.executable.feedTensors) {
    if (t == gg.bias)
      [inputs addObject:graph_data(s, bias, n, 0)];
    else if (t == gg.a)
      [inputs addObject:(ta ? graph_data(s, a, k, m) : graph_data(s, a, m, k))];
    else
      [inputs addObject:(tb ? graph_data(s, b, n, k) : graph_data(s, b, k, n))];
  }
  MPSGraphTensorData* result = graph_data(s, c, m, n);

  commit(s);
  id<MTLCommandBuffer> first = [s.queue commandBufferWithUnretainedReferences];
  if (s.chain_value) [first encodeWaitForEvent:s.chain value:s.chain_value];
  MPSCommandBuffer* mcb = [MPSCommandBuffer commandBufferWithCommandBuffer:first];
  [gg.executable encodeToCommandBuffer:mcb
                           inputsArray:inputs
                          resultsArray:@[ result ]
                   executionDescriptor:nil];
  // MPSGraph may have committed `first` and moved on to another: whichever is
  // current takes the signal and becomes the one fences wait for.
  s.command_buffer = mcb.commandBuffer;
  s.encoder        = nil;
  s.dispatches     = 0;
  commit(s);
  ++s.n_dispatch;
}

void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, int lda,
          const float* b, int ldb, float beta, float* c, int ldc, const float* bias) {
  if (m == 0 || n == 0) return;
  State& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  @autoreleasepool {
    // MPSGraph for the large GEMMs, where its faster kernels win; its cost to
    // encode (~0.2 ms, and a command buffer of its own) loses on small ones.
    // Its tensors are dense: strided operands take MPSMatrixMultiplication.
    // KOKKOS_METAL_GEMM=mps|graph: always one or the other, to compare.
    static const char* mode = getenv("KOKKOS_METAL_GEMM");
    const bool large        = double(m) * n * k >= double(1 << 27);
    const bool graph        = mode ? std::strcmp(mode, "graph") == 0 : large;
    const bool dense        = lda == (ta ? m : k) && ldb == (tb ? k : n) && ldc == n;
    if (graph && beta == 0.0f && dense)
      return gemm_graph(s, ta, tb, m, n, k, alpha, a, lda, b, ldb, c, ldc, bias);
    // MPS has no bias: onto a C it adds to (beta = 1) when it can, else after.
    const bool bias_first = bias && beta == 0.0f;
    if (bias_first) {
      bias_pass(s, c, bias, m, n, ldc, false);
      beta = 1.0f;
    }
    auto matrix = [&](const float* p, int rows, int cols, int ld) {
      auto [buf, off]        = find_buffer(s, p, (size_t(rows - 1) * ld + cols) * sizeof(float));
      MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                     columns:cols
                                                                    rowBytes:ld * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];
      return [[MPSMatrix alloc] initWithBuffer:buf offset:off descriptor:d];
    };
    MPSMatrix* A                          = ta ? matrix(a, k, m, lda) : matrix(a, m, k, lda);
    MPSMatrix* B                          = tb ? matrix(b, n, k, ldb) : matrix(b, k, n, ldb);
    MPSMatrix* C                          = matrix(c, m, n, ldc);
    __strong MPSMatrixMultiplication*& op = s.gemms[{ta, tb, m, n, k, alpha, beta, lda, ldb, ldc}];
    if (!op)
      op = [[MPSMatrixMultiplication alloc] initWithDevice:s.device
                                             transposeLeft:ta
                                            transposeRight:tb
                                                resultRows:m
                                             resultColumns:n
                                           interiorColumns:k
                                                     alpha:alpha
                                                      beta:beta];
    begin(s);
    encoder_barrier(s);
    [op encodeToCommandBuffer:s.command_buffer leftMatrix:A rightMatrix:B resultMatrix:C];
    encoder_barrier(s);
    after_dispatch(s);
    if (bias && !bias_first) bias_pass(s, c, bias, m, n, ldc, true);
    if (s.stats) {
      id<MTLCommandBuffer> cb = s.command_buffer;
      commit(s);
      [cb waitUntilCompleted];
      char label[96];
      std::snprintf(label, sizeof label, "MPS gemm %c%c m=%d n=%d k=%d", ta ? 'T' : 'N',
                    tb ? 'T' : 'N', m, n, k);
      auto& [count, time] = s.per_kernel[label];
      ++count;
      time += cb.GPUEndTime - cb.GPUStartTime;
    }
  }
}

void fence() {
  if (!g_state) return;
  State& s = *g_state;
  id<MTLCommandBuffer> last;
  {
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    commit(s);
    last             = s.last_committed;
    s.last_committed = nil;
  }
  ++s.n_fence;
  if (last) {
    const double t0 = now();
    [last waitUntilCompleted];
    s.t_fence += now() - t0;
    check(last);
  }
  auto* flag = static_cast<uint32_t*>(s.abort_flag.contents);
  if (*flag) {
    *flag = 0;
    fail("Kokkos::abort was called in a kernel");
  }
}

}  // namespace Kokkos::Impl::MetalRuntime

extern "C" void kokkos_metal_register_kernels(const Kokkos::Impl::MetalRuntime::KernelDesc* kernels,
                                              uint32_t count) {
  auto& r = Kokkos::Impl::MetalRuntime::registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  for (uint32_t i = 0; i < count; ++i) r.kernels.emplace(kernels[i].key, kernels[i]);
}
