// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_PARALLEL_RANGE_HPP
#define KOKKOS_METAL_PARALLEL_RANGE_HPP

#include <Kokkos_Parallel.hpp>
#include <Metal/Kokkos_Metal.hpp>

#include <algorithm>
#include <cstring>
#include <type_traits>

// parallel_for, parallel_reduce and parallel_scan over a flat index space.
// A "shape" maps k in [0, size()) to the functor's arguments: RangePolicy
// here, MDRangePolicy in Kokkos_Metal_Parallel_MDRange.hpp.
//
// Each kernel is a Driver struct whose metal_run() is the per-thread body;
// the plugin compiles it for the GPU, and the host enqueues a bitwise copy.

namespace Kokkos::Impl {

template <class Tag, class F, class... Args>
KOKKOS_FORCEINLINE_FUNCTION void metal_invoke(const F& f, Args&&... args) {
  if constexpr (std::is_void_v<Tag>)
    f(static_cast<Args&&>(args)...);
  else
    f(Tag{}, static_cast<Args&&>(args)...);
}

struct MetalThread {
  static uint32_t rank() { return __kmetal_thread_position_in_threadgroup(0); }
  static uint32_t size() { return __kmetal_threads_per_threadgroup(0); }
  static uint32_t group() { return __kmetal_threadgroup_position_in_grid(0); }
  static uint32_t groups() { return __kmetal_threadgroups_per_grid(0); }
  static char* shared() {
    return static_cast<char*>(__kmetal_threadgroup_memory());
  }
};

template <class Index>
struct MetalRangeShape {
  Index begin;
  uint64_t count;

  uint64_t size() const { return count; }
  template <class Tag, class F, class K, class... Args>
  void call(const F& f, K k, Args&&... args) const {
    metal_invoke<Tag>(f, Index(begin + Index(k)), static_cast<Args&&>(args)...);
  }
};

// Threads per threadgroup: the policy's launch bound, else 256, within what
// the compiled kernel allows and what its threadgroup memory fits.
template <class Driver>
uint32_t metal_threads(uint32_t launch_bound, size_t bytes_per_thread = 0) {
  uint32_t t =
      std::min(launch_bound ? launch_bound : 256u, metal_max_threads<Driver>());
  if (bytes_per_thread)
    t = std::min<uint64_t>(
        t, MetalRuntime::max_threadgroup_memory() / bytes_per_thread);
  if (t == 0)
    Kokkos::abort(
        "Kokkos::Metal: a reduction value too large for threadgroup memory");
  uint32_t p = 1;
  while (2 * p <= t) p *= 2;
  return p;
}

template <class Policy>
constexpr uint32_t metal_launch_bound() {
  return Policy::launch_bounds::maxTperB;
}

// Index arithmetic in 32 bits when the index space allows: Apple GPUs do
// 64-bit integer operations in several 32-bit ones. Up to 2^31, k + stride
// cannot wrap (the stride is at most 2^28).
inline bool metal_narrow(uint64_t n) { return n <= (uint64_t(1) << 31); }

// --- parallel_for -----------------------------------------------------------

template <class Shape, class F, class Tag, class K>
struct MetalForDriver {
  F functor;
  Shape shape;

  void metal_run() const {
    const K n      = K(shape.size());
    const K stride = K(MetalThread::groups()) * MetalThread::size();
    for (K k = K(MetalThread::group()) * MetalThread::size() +
               MetalThread::rank();
         k < n; k += stride)
      shape.template call<Tag>(functor, k);
  }
};

template <class Tag, class K, class Shape, class F>
void metal_for_k(const Shape& shape, const F& functor, uint32_t launch_bound) {
  using Driver           = MetalForDriver<Shape, F, Tag, K>;
  const uint64_t n       = shape.size();
  const uint32_t threads = metal_threads<Driver>(launch_bound);
  const uint64_t groups =
      std::min<uint64_t>((n + threads - 1) / threads, 1u << 20);
  metal_launch(Driver{functor, shape}, uint32_t(groups), threads);
}

template <class Tag, class Shape, class F>
void metal_for(const Shape& shape, const F& functor, uint32_t launch_bound) {
  const uint64_t n = shape.size();
  if (n == 0) return;
  if (metal_narrow(n))
    metal_for_k<Tag, uint32_t>(shape, functor, launch_bound);
  else
    metal_for_k<Tag, uint64_t>(shape, functor, launch_bound);
}

// --- reductions
// ---------------------------------------------------------------

template <class Reducer>
using metal_value_t = typename Reducer::value_type;

// Joins the threadgroup's values (one per thread, `size` bytes apart in
// threadgroup memory) into the first.
template <class Reducer>
KOKKOS_FORCEINLINE_FUNCTION void metal_group_reduce(const Reducer& reducer,
                                                    char* values,
                                                    uint32_t size) {
  using V          = metal_value_t<Reducer>;
  const uint32_t t = MetalThread::rank(), n = MetalThread::size();
  uint32_t w = 1;
  while (w < n) w *= 2;
  for (w /= 2; w > 0; w /= 2) {
    __kmetal_threadgroup_barrier();
    if (t < w && t + w < n)
      reducer.join(reinterpret_cast<V*>(values + t * size),
                   reinterpret_cast<const V*>(values + (t + w) * size));
  }
  __kmetal_threadgroup_barrier();
}

// Each thread reduces a grid-strided set of indices; each group its threads;
// partials[group] holds the result.
template <class Shape, class FR, class Tag, class K>
struct MetalReduceDriver {
  FR fr;
  Shape shape;
  char* partials;

  void metal_run() const {
    using R             = typename FR::reducer_type;
    using V             = metal_value_t<R>;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    const uint32_t t    = MetalThread::rank();
    char* values        = MetalThread::shared();
    V* mine             = reinterpret_cast<V*>(values + t * size);
    const K n           = K(shape.size());
    const K stride      = K(MetalThread::groups()) * MetalThread::size();
    const K first       = K(MetalThread::group()) * MetalThread::size() + t;
    if constexpr (R::static_value_size() > 0) {
      // Accumulate in registers, not threadgroup memory.
      V v;
      reducer.init(&v);
      for (K k = first; k < n; k += stride)
        shape.template call<Tag>(fr.get_functor(), k, reducer.reference(&v));
      *mine = v;
    } else {
      reducer.init(mine);
      for (K k = first; k < n; k += stride)
        shape.template call<Tag>(fr.get_functor(), k, reducer.reference(mine));
    }
    metal_group_reduce(reducer, values, size);
    if (t == 0)
      reducer.copy(reinterpret_cast<V*>(partials + MetalThread::group() * size),
                   reinterpret_cast<V*>(values));
  }
};

// One threadgroup joins the partials, applies final() and stores the result.
template <class FR>
struct MetalReduceFinal {
  FR fr;
  const char* partials;
  uint32_t count;
  char* result;

  void metal_run() const {
    using R             = typename FR::reducer_type;
    using V             = metal_value_t<R>;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    const uint32_t t    = MetalThread::rank();
    char* values        = MetalThread::shared();
    V* mine             = reinterpret_cast<V*>(values + t * size);
    reducer.init(mine);
    for (uint32_t k = t; k < count; k += MetalThread::size())
      reducer.join(mine, reinterpret_cast<const V*>(partials + k * size));
    metal_group_reduce(reducer, values, size);
    if (t == 0) {
      reducer.final(reinterpret_cast<V*>(values));
      reducer.copy(reinterpret_cast<V*>(result), reinterpret_cast<V*>(values));
    }
  }
};

inline size_t metal_align(size_t n) { return (n + 15) / 16 * 16; }

// The result goes straight to `result` when a kernel can write it, else
// through scratch memory and a fence.
template <class Tag, class K, class Shape, class FR>
void metal_reduce_k(const Shape& shape, const FR& fr,
                    typename FR::reducer_type::pointer_type result,
                    bool result_on_device, uint32_t launch_bound) {
  using Driver           = MetalReduceDriver<Shape, FR, Tag, K>;
  using Final            = MetalReduceFinal<FR>;
  const uint32_t size    = fr.get_reducer().value_size();
  const uint32_t threads = metal_threads<Driver>(launch_bound, size);
  const uint64_t n       = shape.size();
  const uint32_t groups =
      uint32_t(std::clamp<uint64_t>((n + threads - 1) / threads, 1, 1024));

  char* scratch =
      static_cast<char*>(metal_scratch(metal_align(groups * size) + size));
  char* out = result_on_device && result ? reinterpret_cast<char*>(result)
                                         : scratch + metal_align(groups * size);
  metal_launch(Driver{fr, shape, scratch}, groups, threads, threads * size);
  const uint32_t final_threads = metal_threads<Final>(0, size);
  metal_launch(Final{fr, scratch, groups, out}, 1, final_threads,
               final_threads * size);
  if (result && !result_on_device) {
    Kokkos::Experimental::Metal().fence(
        "Kokkos::Metal::parallel_reduce: result to host");
    std::memcpy(result, out, size);
  }
}

template <class Tag, class Shape, class FR>
void metal_reduce(const Shape& shape, const FR& fr,
                  typename FR::reducer_type::pointer_type result,
                  bool result_on_device, uint32_t launch_bound) {
  if (metal_narrow(shape.size()))
    metal_reduce_k<Tag, uint32_t>(shape, fr, result, result_on_device,
                                  launch_bound);
  else
    metal_reduce_k<Tag, uint64_t>(shape, fr, result, result_on_device,
                                  launch_bound);
}

// --- scans
// --------------------------------------------------------------------
//
// Group g scans a contiguous block of indices, thread t a contiguous chunk of
// that. Up: the groups' sums. Partials: their exclusive scan, by one thread.
// Down: each group scans its threads' sums from its partial and each thread
// calls the functor with final = true along its chunk.

template <class Shape, class FR, class Tag, class K>
struct MetalScanUp {
  FR fr;
  Shape shape;
  char* partials;
  uint64_t chunk;

  void metal_run() const {
    using R             = typename FR::reducer_type;
    using V             = metal_value_t<R>;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    const uint32_t t    = MetalThread::rank();
    char* values        = MetalThread::shared();
    V* mine             = reinterpret_cast<V*>(values + t * size);
    const K lo = (K(MetalThread::group()) * MetalThread::size() + t) * K(chunk);
    const K hi = Kokkos::min(K(lo + chunk), K(shape.size()));
    reducer.init(mine);
    for (K k = lo; k < hi; ++k)
      shape.template call<Tag>(fr.get_functor(), k, reducer.reference(mine),
                               false);
    metal_group_reduce(reducer, values, size);
    if (t == 0)
      reducer.copy(reinterpret_cast<V*>(partials + MetalThread::group() * size),
                   reinterpret_cast<V*>(values));
  }
};

template <class FR>
struct MetalScanPartials {
  FR fr;
  char* partials;
  uint32_t count;
  char* total;

  void metal_run() const {
    using R = typename FR::reducer_type;
    using V = metal_value_t<R>;
    if (MetalThread::rank() != 0) return;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    V* run              = reinterpret_cast<V*>(MetalThread::shared());
    V* tmp              = reinterpret_cast<V*>(MetalThread::shared() + size);
    reducer.init(run);
    for (uint32_t g = 0; g < count; ++g) {
      V* p = reinterpret_cast<V*>(partials + g * size);
      reducer.copy(tmp, p);
      reducer.copy(p, run);
      reducer.join(run, tmp);
    }
    if (total) reducer.copy(reinterpret_cast<V*>(total), run);
  }
};

template <class Shape, class FR, class Tag, class K>
struct MetalScanDown {
  FR fr;
  Shape shape;
  const char* partials;
  uint64_t chunk;

  void metal_run() const {
    using R             = typename FR::reducer_type;
    using V             = metal_value_t<R>;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    const uint32_t t = MetalThread::rank(), n = MetalThread::size();
    char* values = MetalThread::shared();
    V* mine      = reinterpret_cast<V*>(values + t * size);
    const K lo = (K(MetalThread::group()) * MetalThread::size() + t) * K(chunk);
    const K hi = Kokkos::min(K(lo + chunk), K(shape.size()));
    reducer.init(mine);
    for (K k = lo; k < hi; ++k)
      shape.template call<Tag>(fr.get_functor(), k, reducer.reference(mine),
                               false);
    __kmetal_threadgroup_barrier();
    if (t == 0) {
      V* run = reinterpret_cast<V*>(values + n * size);
      V* tmp = reinterpret_cast<V*>(values + (n + 1) * size);
      reducer.copy(run, reinterpret_cast<const V*>(
                            partials + MetalThread::group() * size));
      for (uint32_t u = 0; u < n; ++u) {
        V* s = reinterpret_cast<V*>(values + u * size);
        reducer.copy(tmp, s);
        reducer.copy(s, run);
        reducer.join(run, tmp);
      }
    }
    __kmetal_threadgroup_barrier();
    for (K k = lo; k < hi; ++k)
      shape.template call<Tag>(fr.get_functor(), k, reducer.reference(mine),
                               true);
  }
};

template <class Tag, class K, class Shape, class FR>
void metal_scan_k(const Shape& shape, const FR& fr,
                  typename FR::reducer_type::pointer_type total,
                  bool total_on_device, uint32_t launch_bound) {
  using Up            = MetalScanUp<Shape, FR, Tag, K>;
  using Partials      = MetalScanPartials<FR>;
  using Down          = MetalScanDown<Shape, FR, Tag, K>;
  const uint32_t size = fr.get_reducer().value_size();
  const uint64_t n    = shape.size();
  uint32_t threads    = std::min(metal_threads<Up>(launch_bound, size),
                                 metal_threads<Down>(launch_bound, size));
  // Down keeps two more values than it has threads.
  while (threads > 1 &&
         (threads + 2) * size > MetalRuntime::max_threadgroup_memory())
    threads /= 2;
  const uint32_t groups =
      uint32_t(std::clamp<uint64_t>((n + threads - 1) / threads, 1, 1024));
  const uint64_t chunk = std::max<uint64_t>(
      1, (n + uint64_t(groups) * threads - 1) / (uint64_t(groups) * threads));

  char* scratch =
      static_cast<char*>(metal_scratch(metal_align(groups * size) + size));
  char* out = total_on_device && total ? reinterpret_cast<char*>(total)
                                       : scratch + metal_align(groups * size);
  metal_launch(Up{fr, shape, scratch, chunk}, groups, threads, threads * size);
  metal_launch(Partials{fr, scratch, groups, out}, 1, 1, 2 * size);
  metal_launch(Down{fr, shape, scratch, chunk}, groups, threads,
               (threads + 2) * size);
  if (total && !total_on_device) {
    Kokkos::Experimental::Metal().fence(
        "Kokkos::Metal::parallel_scan: total to host");
    std::memcpy(total, out, size);
  }
}

template <class Tag, class Shape, class FR>
void metal_scan(const Shape& shape, const FR& fr,
                typename FR::reducer_type::pointer_type total,
                bool total_on_device, uint32_t launch_bound) {
  // The chunks' ends reach n + chunk at most.
  if (metal_narrow(2 * shape.size()))
    metal_scan_k<Tag, uint32_t>(shape, fr, total, total_on_device,
                                launch_bound);
  else
    metal_scan_k<Tag, uint64_t>(shape, fr, total, total_on_device,
                                launch_bound);
}

// --- RangePolicy
// --------------------------------------------------------------

template <class Policy>
MetalRangeShape<typename Policy::member_type> metal_range_shape(
    const Policy& policy) {
  return {policy.begin(), uint64_t(policy.end() - policy.begin())};
}

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Experimental::Metal> {
  using Policy = Kokkos::RangePolicy<Traits...>;
  FunctorType m_functor;
  Policy m_policy;

 public:
  ParallelFor(const FunctorType& functor, const Policy& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    metal_for<typename Policy::work_tag>(metal_range_shape(m_policy), m_functor,
                                         metal_launch_bound<Policy>());
  }
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, Kokkos::RangePolicy<Traits...>,
                     Kokkos::Experimental::Metal> {
  using Policy = Kokkos::RangePolicy<Traits...>;
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
        metal_range_shape(m_policy), m_functor_reducer, m_result,
        m_result_on_device, metal_launch_bound<Policy>());
  }
};

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Experimental::Metal> {
  using Policy = Kokkos::RangePolicy<Traits...>;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType, void>;
  using FR = CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>;
  FR m_functor_reducer;
  Policy m_policy;

 public:
  ParallelScan(const FunctorType& functor, const Policy& policy)
      : m_functor_reducer(functor, typename Analysis::Reducer{functor}),
        m_policy(policy) {}

  void execute() const {
    metal_scan<typename Policy::work_tag>(metal_range_shape(m_policy),
                                          m_functor_reducer, nullptr, false,
                                          metal_launch_bound<Policy>());
  }
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Experimental::Metal> {
  using Policy   = Kokkos::RangePolicy<Traits...>;
  using Analysis = FunctorAnalysis<FunctorPatternInterface::SCAN, Policy,
                                   FunctorType, ReturnType>;
  using FR = CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>;
  using pointer_type = typename Analysis::pointer_type;
  FR m_functor_reducer;
  Policy m_policy;
  pointer_type m_result;
  bool m_result_on_device;

 public:
  template <class ViewType>
  ParallelScanWithTotal(const FunctorType& functor, const Policy& policy,
                        const ViewType& result)
      : m_functor_reducer(functor, typename Analysis::Reducer{functor}),
        m_policy(policy),
        m_result(result.data()),
        m_result_on_device(
            MemorySpaceAccess<Kokkos::Experimental::MetalSpace,
                              typename ViewType::memory_space>::accessible) {}

  void execute() const {
    metal_scan<typename Policy::work_tag>(
        metal_range_shape(m_policy), m_functor_reducer, m_result,
        m_result_on_device, metal_launch_bound<Policy>());
  }
};

}  // namespace Kokkos::Impl

#endif
