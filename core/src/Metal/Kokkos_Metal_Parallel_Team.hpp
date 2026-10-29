// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_METAL_PARALLEL_TEAM_HPP
#define KOKKOS_METAL_PARALLEL_TEAM_HPP

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Metal/Kokkos_Metal_Parallel_Range.hpp>

#include <algorithm>
#include <sstream>

// TeamPolicy. A team is one threadgroup of team_size * vector_length threads,
// thread t being team rank t / vector_length and vector lane
// t % vector_length; vector lengths are powers of two up to the SIMD width,
// so a thread's lanes share a SIMD group. Threadgroups take league ranks in a
// grid-strided loop.
//
// Threadgroup memory: kMetalTeamReserve bytes of slots for team collectives,
// then the level-0 scratch. Level-1 scratch is device memory, a slice per
// threadgroup.

namespace Kokkos::Impl {

inline constexpr uint32_t kMetalTeamReserve = 2048;
// Most threadgroups a team kernel launches: what level-1 scratch is sized for.
inline constexpr uint32_t kMetalMaxTeamGroups = 4096;

// Shuffles of any trivially copyable T, a 32-bit word at a time.
template <class T, class Shuffle>
KOKKOS_FORCEINLINE_FUNCTION T metal_shuffle_words(const T& v, Shuffle shuffle) {
  constexpr int n = (sizeof(T) + 3) / 4;
  uint32_t w[n]   = {};
  __builtin_memcpy(w, &v, sizeof(T));
  for (int i = 0; i < n; ++i) w[i] = shuffle(w[i]);
  T r;
  __builtin_memcpy(&r, w, sizeof(T));
  return r;
}
template <class T>
KOKKOS_FORCEINLINE_FUNCTION T metal_shuffle(const T& v, uint32_t lane) {
  return metal_shuffle_words(
      v, [=](uint32_t x) { return __kmetal_simd_shuffle(x, lane); });
}
template <class T>
KOKKOS_FORCEINLINE_FUNCTION T metal_shuffle_down(const T& v, uint32_t delta) {
  return metal_shuffle_words(
      v, [=](uint32_t x) { return __kmetal_simd_shuffle_down(x, delta); });
}
template <class T>
KOKKOS_FORCEINLINE_FUNCTION T metal_shuffle_up(const T& v, uint32_t delta) {
  return metal_shuffle_words(
      v, [=](uint32_t x) { return __kmetal_simd_shuffle_up(x, delta); });
}

class MetalTeamMember {
 public:
  using execution_space      = Kokkos::Experimental::Metal;
  using scratch_memory_space = execution_space::scratch_memory_space;
  using team_handle          = MetalTeamMember;

 private:
  char* m_slots;  // kMetalTeamReserve bytes of threadgroup memory
  scratch_memory_space m_scratch;
  int m_league_rank;
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  int m_thread;  // index in the threadgroup

 public:
  KOKKOS_INLINE_FUNCTION
  MetalTeamMember(char* slots, void* scratch0, size_t scratch0_size,
                  void* scratch1, size_t scratch1_size, int league_rank,
                  int league_size, int team_size, int vector_length)
      : m_slots(slots),
        m_scratch(scratch0, scratch0_size, scratch1, scratch1_size),
        m_league_rank(league_rank),
        m_league_size(league_size),
        m_team_size(team_size),
        m_vector_length(vector_length),
        m_thread(int(__kmetal_thread_position_in_threadgroup(0))) {}

  KOKKOS_INLINE_FUNCTION const scratch_memory_space& team_shmem() const {
    return m_scratch.set_team_thread_mode(0, 1, 0);
  }
  KOKKOS_INLINE_FUNCTION const scratch_memory_space& team_scratch(
      const int& level) const {
    return m_scratch.set_team_thread_mode(level, 1, 0);
  }
  KOKKOS_INLINE_FUNCTION const scratch_memory_space& thread_scratch(
      const int& level) const {
    return m_scratch.set_team_thread_mode(level, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const {
    return m_thread / m_vector_length;
  }
  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }
  KOKKOS_INLINE_FUNCTION int impl_vector_length() const {
    return m_vector_length;
  }
  KOKKOS_INLINE_FUNCTION int impl_vector_lane() const {
    return m_thread % m_vector_length;
  }
  KOKKOS_INLINE_FUNCTION char* impl_slots() const { return m_slots; }

  KOKKOS_INLINE_FUNCTION void team_barrier() const {
    __kmetal_threadgroup_barrier();
  }

  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType& val,
                                             const int& thread_id) const {
    static_assert(sizeof(ValueType) <= kMetalTeamReserve,
                  "Kokkos::Metal: team_broadcast value too large");
    __kmetal_threadgroup_barrier();
    if (team_rank() == thread_id && impl_vector_lane() == 0)
      __builtin_memcpy(m_slots, &val, sizeof(ValueType));
    __kmetal_threadgroup_barrier();
    __builtin_memcpy(&val, m_slots, sizeof(ValueType));
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(Closure const& f, ValueType& val,
                                             const int& thread_id) const {
    if (team_rank() == thread_id) f(val);
    team_broadcast(val, thread_id);
  }

  // Joins `value` over the team. With stride = vector length, over team
  // ranks (a thread's lanes hold the same value); with stride 1, over every
  // thread. Every thread gets the result.
  template <class T, class Join>
  KOKKOS_INLINE_FUNCTION T impl_reduce(T v, const Join& join,
                                       uint32_t stride) const {
    const uint32_t width   = __kmetal_threads_per_simdgroup();
    const uint32_t lane    = __kmetal_thread_index_in_simdgroup();
    const uint32_t sg      = __kmetal_simdgroup_index_in_threadgroup();
    const uint32_t threads = uint32_t(m_team_size * m_vector_length);
    const uint32_t active  = Kokkos::min(width, threads - sg * width);
    for (uint32_t o = width / 2; o >= stride; o /= 2) {
      T other = metal_shuffle_down(v, o);
      if (lane + o < active) join(v, other);
    }
    const uint32_t groups = (threads + width - 1) / width;
    if (groups == 1) return metal_shuffle(v, lane % stride);
    static_assert(32 * sizeof(T) <= kMetalTeamReserve,
                  "Kokkos::Metal: team reduction value too large");
    T* slots = reinterpret_cast<T*>(m_slots);
    __kmetal_threadgroup_barrier();
    if (lane == 0) slots[sg] = v;
    __kmetal_threadgroup_barrier();
    T r = slots[0];
    for (uint32_t s = 1; s < groups; ++s) join(r, slots[s]);
    return r;
  }

  template <class ReducerType>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
  team_reduce(ReducerType const& reducer) const noexcept {
    team_reduce(reducer, reducer.reference());
  }

  template <class ReducerType>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
  team_reduce(ReducerType const& reducer,
              typename ReducerType::value_type& value) const noexcept {
    using V = typename ReducerType::value_type;
    value   = impl_reduce(
        value, [&](V& a, const V& b) { reducer.join(a, b); },
        uint32_t(m_vector_length));
  }

  // Exclusive scan over team ranks, in team-rank order; with `carry`, the
  // team's total is added to it atomically and its old value to the result.
  template <class T>
  KOKKOS_INLINE_FUNCTION T team_scan(const T& value,
                                     T* const carry = nullptr) const {
    T* slots = reinterpret_cast<T*>(m_slots);
    if (sizeof(T) * (m_team_size + 1) > kMetalTeamReserve)
      Kokkos::abort("Kokkos::Metal: team_scan over too many team ranks");
    __kmetal_threadgroup_barrier();
    if (impl_vector_lane() == 0) slots[team_rank()] = value;
    __kmetal_threadgroup_barrier();
    if (m_thread == 0) {
      T total = T(0);
      for (int r = 0; r < m_team_size; ++r) total += slots[r];
      T run = carry ? Kokkos::atomic_fetch_add(carry, total) : T(0);
      for (int r = 0; r < m_team_size; ++r) {
        T x      = slots[r];
        slots[r] = run;
        run += x;
      }
    }
    __kmetal_threadgroup_barrier();
    return slots[team_rank()];
  }
};

template <typename iType>
struct TeamThreadRangeBoundariesStruct<iType, MetalTeamMember> {
  using index_type = iType;
  const MetalTeamMember& member;
  const iType start;
  const iType end;

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const MetalTeamMember& m, iType count)
      : member(m), start(0), end(count) {}
  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const MetalTeamMember& m, iType b, iType e)
      : member(m), start(b), end(e) {}
};

template <typename iType>
struct TeamVectorRangeBoundariesStruct<iType, MetalTeamMember> {
  using index_type = iType;
  const MetalTeamMember& member;
  const iType start;
  const iType end;

  KOKKOS_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const MetalTeamMember& m, const iType& count)
      : member(m), start(0), end(count) {}
  KOKKOS_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const MetalTeamMember& m, const iType& b,
                                  const iType& e)
      : member(m), start(b), end(e) {}
};

template <typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, MetalTeamMember> {
  using index_type = iType;
  const MetalTeamMember& member;
  const index_type start;
  const index_type end;

  KOKKOS_INLINE_FUNCTION
  ThreadVectorRangeBoundariesStruct(const MetalTeamMember& m, index_type count)
      : member(m), start(0), end(count) {}
  KOKKOS_INLINE_FUNCTION
  ThreadVectorRangeBoundariesStruct(const MetalTeamMember& m, index_type b,
                                    index_type e)
      : member(m), start(b), end(e) {}
};

}  // namespace Kokkos::Impl

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::MetalTeamMember>
    TeamThreadRange(const Impl::MetalTeamMember& m, iType count) {
  return {m, count};
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, Impl::MetalTeamMember>
TeamThreadRange(const Impl::MetalTeamMember& m, iType1 begin, iType2 end) {
  using iType = std::common_type_t<iType1, iType2>;
  return {m, iType(begin), iType(end)};
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>
    TeamVectorRange(const Impl::MetalTeamMember& m, const iType& count) {
  return {m, count};
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamVectorRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, Impl::MetalTeamMember>
TeamVectorRange(const Impl::MetalTeamMember& m, const iType1& begin,
                const iType2& end) {
  using iType = std::common_type_t<iType1, iType2>;
  return {m, iType(begin), iType(end)};
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>
    ThreadVectorRange(const Impl::MetalTeamMember& m, iType count) {
  return {m, count};
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::ThreadVectorRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, Impl::MetalTeamMember>
ThreadVectorRange(const Impl::MetalTeamMember& m, iType1 begin, iType2 end) {
  using iType = std::common_type_t<iType1, iType2>;
  return {m, iType(begin), iType(end)};
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::MetalTeamMember> PerTeam(
    const Impl::MetalTeamMember& m) {
  return Impl::ThreadSingleStruct<Impl::MetalTeamMember>(m);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::MetalTeamMember> PerThread(
    const Impl::MetalTeamMember& m) {
  return Impl::VectorSingleStruct<Impl::MetalTeamMember>(m);
}

// --- nested parallel_for
// ------------------------------------------------------

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure) {
  for (iType i = r.start + r.member.team_rank(); i < r.end;
       i += r.member.team_size())
    closure(i);
}

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure) {
  const int v = r.member.impl_vector_length();
  for (iType i =
           r.start + r.member.team_rank() * v + r.member.impl_vector_lane();
       i < r.end; i += r.member.team_size() * v)
    closure(i);
}

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure) {
  for (iType i = r.start + r.member.impl_vector_lane(); i < r.end;
       i += r.member.impl_vector_length())
    closure(i);
}

// --- nested parallel_reduce
// ---------------------------------------------------

namespace Impl {

// The value joined over the vector lanes of one thread; every lane gets it.
template <class T, class Join>
KOKKOS_INLINE_FUNCTION T metal_vector_reduce(const MetalTeamMember& m, T v,
                                             const Join& join) {
  const uint32_t n = m.impl_vector_length();
  if (n == 1) return v;
  const uint32_t lane = __kmetal_thread_index_in_simdgroup();
  const uint32_t my   = lane % n;
  for (uint32_t o = n / 2; o > 0; o /= 2) {
    T other = metal_shuffle_down(v, o);
    if (my + o < n) join(v, other);
  }
  return metal_shuffle(v, lane - my);
}

template <class Reducer, class Loop>
KOKKOS_INLINE_FUNCTION void metal_nested_reduce(const MetalTeamMember& m,
                                                const Reducer& reducer,
                                                uint32_t stride, bool vector,
                                                const Loop& loop) {
  using V = typename Reducer::value_type;
  V v;
  reducer.init(v);
  loop(v);
  auto join = [&](V& a, const V& b) { reducer.join(a, b); };
  reducer.reference() =
      vector ? metal_vector_reduce(m, v, join) : m.impl_reduce(v, join, stride);
}

}  // namespace Impl

template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, const ReducerType& reducer) {
  Impl::metal_nested_reduce(r.member, reducer, r.member.impl_vector_length(),
                            false, [&](auto& v) {
                              for (iType i = r.start + r.member.team_rank();
                                   i < r.end; i += r.member.team_size())
                                closure(i, v);
                            });
}

template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, const ReducerType& reducer) {
  const int n = r.member.impl_vector_length();
  Impl::metal_nested_reduce(r.member, reducer, 1, false, [&](auto& v) {
    for (iType i =
             r.start + r.member.team_rank() * n + r.member.impl_vector_lane();
         i < r.end; i += r.member.team_size() * n)
      closure(i, v);
  });
}

template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, const ReducerType& reducer) {
  Impl::metal_nested_reduce(r.member, reducer, 1, true, [&](auto& v) {
    for (iType i = r.start + r.member.impl_vector_lane(); i < r.end;
         i += r.member.impl_vector_length())
      closure(i, v);
  });
}

template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, ValueType& result) {
  parallel_reduce(r, closure, Kokkos::Sum<ValueType>(result));
}

template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, ValueType& result) {
  parallel_reduce(r, closure, Kokkos::Sum<ValueType>(result));
}

template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::MetalTeamMember>& r,
                const Closure& closure, ValueType& result) {
  parallel_reduce(r, closure, Kokkos::Sum<ValueType>(result));
}

// --- nested parallel_scan
// -----------------------------------------------------

// Blocks of team_size indices: each team rank its index's contribution, a
// team scan for the prefixes, then the final pass.
template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure, ValueType& return_val) {
  const auto& m = r.member;
  ValueType carry{};
  ValueType* slots = reinterpret_cast<ValueType*>(m.impl_slots());
  for (iType base = r.start; base < r.end; base += m.team_size()) {
    const iType i = base + m.team_rank();
    ValueType v{};
    if (i < r.end) closure(i, v, false);
    ValueType prefix = m.team_scan(v);
    // The block's total: the last rank's prefix plus its own value.
    if (m.team_rank() == m.team_size() - 1 && m.impl_vector_lane() == 0)
      slots[m.team_size()] = prefix + v;
    m.team_barrier();
    ValueType total = slots[m.team_size()];
    prefix += carry;
    if (i < r.end) closure(i, prefix, true);
    carry += total;
  }
  return_val = carry;
}

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure) {
  using Analysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                            TeamPolicy<Kokkos::Experimental::Metal>, Closure,
                            void>;
  typename Analysis::value_type total;
  parallel_scan(r, closure, total);
}

// Blocks of vector_length indices: an inclusive scan across the lanes by
// shuffles, shifted for the exclusive prefix.
template <typename iType, class Closure, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_scan(const Impl::ThreadVectorRangeBoundariesStruct<
                  iType, Impl::MetalTeamMember>& r,
              const Closure& closure, const ReducerType& reducer) {
  using V             = typename ReducerType::value_type;
  const auto& m       = r.member;
  const uint32_t n    = m.impl_vector_length();
  const uint32_t my   = m.impl_vector_lane();
  const uint32_t lane = __kmetal_thread_index_in_simdgroup();
  V carry;
  reducer.init(carry);
  for (iType base = r.start; base < r.end; base += n) {
    const iType i = base + my;
    V v;
    reducer.init(v);
    if (i < r.end) closure(i, v, false);
    V inclusive = v;
    for (uint32_t o = 1; o < n; o *= 2) {
      V other = Impl::metal_shuffle_up(inclusive, o);
      if (my >= o) reducer.join(inclusive, other);
    }
    V exclusive = Impl::metal_shuffle_up(inclusive, 1);
    if (my == 0) reducer.init(exclusive);
    V prefix = carry;
    reducer.join(prefix, exclusive);
    if (i < r.end) closure(i, prefix, true);
    V total = Impl::metal_shuffle(inclusive, lane - my + n - 1);
    reducer.join(carry, total);
  }
  reducer.reference() = carry;
}

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure) {
  using Analysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN,
                            TeamPolicy<Kokkos::Experimental::Metal>, Closure,
                            void>;
  using V = typename Analysis::value_type;
  V total;
  parallel_scan(r, closure, Kokkos::Sum<V>(total));
}

template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::MetalTeamMember>&
        r,
    const Closure& closure, ValueType& return_val) {
  ValueType total;
  parallel_scan(r, closure, Kokkos::Sum<ValueType>(total));
  return_val = total;
}

// --- single -----------------------------------------------------------------

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::MetalTeamMember>& s,
    const FunctorType& lambda) {
  if (s.team_member.impl_vector_lane() == 0) lambda();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::MetalTeamMember>& s,
    const FunctorType& lambda) {
  if (s.team_member.team_rank() == 0 && s.team_member.impl_vector_lane() == 0)
    lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::MetalTeamMember>& s,
    const FunctorType& lambda, ValueType& val) {
  const auto& m = s.team_member;
  if (m.impl_vector_lane() == 0) lambda(val);
  const uint32_t lane = __kmetal_thread_index_in_simdgroup();
  val = Impl::metal_shuffle(val, lane - uint32_t(m.impl_vector_lane()));
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::MetalTeamMember>& s,
    const FunctorType& lambda, ValueType& val) {
  if (s.team_member.team_rank() == 0 && s.team_member.impl_vector_lane() == 0)
    lambda(val);
  s.team_member.team_broadcast(val, 0);
}

}  // namespace Kokkos

// --- the policy and its kernels
// -----------------------------------------------

namespace Kokkos::Impl {

// What a team kernel needs of its policy, on the device.
struct MetalTeamShape {
  int league_size;
  int team_size;
  int vector_length;
  uint32_t scratch0;  // level-0 bytes per team
  char* scratch1;     // level-1 memory, scratch1_size bytes per threadgroup
  uint64_t scratch1_size;

  template <class F>
  KOKKOS_INLINE_FUNCTION void for_each_league(const F& f) const {
    char* tg            = MetalThread::shared();
    const uint32_t g    = MetalThread::group();
    const uint32_t step = MetalThread::groups();
    for (int league = int(g); league < league_size; league += int(step)) {
      MetalTeamMember member(tg, tg + kMetalTeamReserve, scratch0,
                             scratch1 + g * scratch1_size, scratch1_size,
                             league, league_size, team_size, vector_length);
      f(member);
      // The next league reuses the scratch memory.
      if (league + int(step) < league_size) __kmetal_threadgroup_barrier();
    }
  }
};

template <class F, class Tag>
struct MetalTeamForDriver {
  F functor;
  MetalTeamShape shape;

  void metal_run() const {
    shape.for_each_league(
        [&](const MetalTeamMember& m) { metal_invoke<Tag>(functor, m); });
  }
};

// Each thread joins its leagues' values; the threadgroup its threads, in
// the threadgroup memory after the scratch; partials[group] the result.
template <class FR, class Tag>
struct MetalTeamReduceDriver {
  FR fr;
  MetalTeamShape shape;
  char* partials;

  void metal_run() const {
    using R             = typename FR::reducer_type;
    using V             = metal_value_t<R>;
    const R& reducer    = fr.get_reducer();
    const uint32_t size = reducer.value_size();
    const uint32_t t    = MetalThread::rank();
    char* values = MetalThread::shared() + kMetalTeamReserve + shape.scratch0;
    V* mine      = reinterpret_cast<V*>(values + t * size);
    if constexpr (R::static_value_size() > 0) {
      V v;
      reducer.init(&v);
      shape.for_each_league([&](const MetalTeamMember& m) {
        metal_invoke<Tag>(fr.get_functor(), m, reducer.reference(&v));
      });
      *mine = v;
    } else {
      reducer.init(mine);
      shape.for_each_league([&](const MetalTeamMember& m) {
        metal_invoke<Tag>(fr.get_functor(), m, reducer.reference(mine));
      });
    }
    metal_group_reduce(reducer, values, size);
    if (t == 0)
      reducer.copy(reinterpret_cast<V*>(partials + MetalThread::group() * size),
                   reinterpret_cast<V*>(values));
  }
};

void* metal_team_scratch(size_t bytes);

template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::Metal, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  using execution_policy = TeamPolicyInternal;
  using traits           = PolicyTraits<Properties...>;
  using execution_space  = Kokkos::Experimental::Metal;
  using member_type      = MetalTeamMember;

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

 private:
  execution_space m_space;
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];
  int m_chunk_size;
  bool m_tune_team_size;
  bool m_tune_vector_length;

  // The most threads per threadgroup, all limits but the kernel's own.
  int max_threads_for(size_t value_size) const {
    const size_t mem = MetalRuntime::max_threadgroup_memory() -
                       kMetalTeamReserve - m_team_scratch_size[0];
    // Per-thread scratch is per team rank: shared by its vector lanes.
    const size_t per_thread =
        m_thread_scratch_size[0] / m_vector_length + value_size;
    size_t t = per_thread ? mem / per_thread : 1024;
    t        = std::min<size_t>(t, 1024);
    if (traits::launch_bounds::maxTperB)
      t = std::min<size_t>(t, traits::launch_bounds::maxTperB);
    return int(t);
  }

 public:
  template <class... OtherProperties>
  TeamPolicyInternal(TeamPolicyInternal<OtherProperties...> const& p)
      : m_space(p.m_space),
        m_league_size(p.m_league_size),
        m_team_size(p.m_team_size),
        m_vector_length(p.m_vector_length),
        m_team_scratch_size{p.m_team_scratch_size[0], p.m_team_scratch_size[1]},
        m_thread_scratch_size{p.m_thread_scratch_size[0],
                              p.m_thread_scratch_size[1]},
        m_chunk_size(p.m_chunk_size),
        m_tune_team_size(p.m_tune_team_size),
        m_tune_vector_length(p.m_tune_vector_length) {}

  TeamPolicyInternal()
      : m_league_size(0),
        m_team_size(-1),
        m_vector_length(1),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(32),
        m_tune_team_size(false),
        m_tune_vector_length(false) {}

  TeamPolicyInternal(execution_space space, int league_size,
                     int team_size_request, int vector_length_request = 1)
      : m_space(std::move(space)),
        m_league_size(league_size),
        m_team_size(team_size_request),
        m_vector_length(impl_determine_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(32),
        m_tune_team_size(team_size_request <= 0),
        m_tune_vector_length(vector_length_request <= 0) {
    if (m_team_size > 0 && m_team_size * m_vector_length > 1024) {
      std::stringstream error;
      error << "Kokkos::TeamPolicy<Metal>: requested too large team size "
            << m_team_size << ", maximum " << 1024 / m_vector_length;
      Kokkos::Impl::throw_runtime_exception(error.str());
    }
  }
  TeamPolicyInternal(execution_space space, int league_size, const AUTO_t&,
                     int vector_length_request = 1)
      : TeamPolicyInternal(std::move(space), league_size, -1,
                           vector_length_request) {}
  TeamPolicyInternal(execution_space space, int league_size,
                     int team_size_request, const AUTO_t&)
      : TeamPolicyInternal(std::move(space), league_size, team_size_request,
                           -1) {}
  TeamPolicyInternal(execution_space space, int league_size, const AUTO_t&,
                     const AUTO_t&)
      : TeamPolicyInternal(std::move(space), league_size, -1, -1) {}
  TeamPolicyInternal(int league_size, int team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(execution_space(), league_size, team_size_request,
                           vector_length_request) {}
  TeamPolicyInternal(int league_size, const AUTO_t&,
                     int vector_length_request = 1)
      : TeamPolicyInternal(execution_space(), league_size, -1,
                           vector_length_request) {}
  TeamPolicyInternal(int league_size, int team_size_request, const AUTO_t&)
      : TeamPolicyInternal(execution_space(), league_size, team_size_request,
                           -1) {}
  TeamPolicyInternal(int league_size, const AUTO_t&, const AUTO_t&)
      : TeamPolicyInternal(execution_space(), league_size, -1, -1) {}
  TeamPolicyInternal(const PolicyUpdate, const TeamPolicyInternal& other,
                     execution_space space)
      : TeamPolicyInternal(other) {
    m_space = std::move(space);
  }

  static int vector_length_max() { return 32; }
  static int impl_determine_vector_length(int requested) {
    return int(Kokkos::bit_floor(
        unsigned(std::clamp(requested, 1, vector_length_max()))));
  }
  static int scratch_size_max(int level) {
    return level == 0
               ? int(MetalRuntime::max_threadgroup_memory() - kMetalTeamReserve)
               : 256 * 1024 * 1024 / int(kMetalMaxTeamGroups);
  }

  template <class FunctorType>
  int team_size_max(const FunctorType&, const ParallelForTag&) const {
    using Driver = MetalTeamForDriver<FunctorType, typename traits::work_tag>;
    return std::min<int>(metal_max_threads<Driver>(), max_threads_for(0)) /
           m_vector_length;
  }
  template <class FunctorType>
  int team_size_max(const FunctorType& f, const ParallelReduceTag&) const {
    using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                     TeamPolicyInternal, FunctorType, void>;
    return team_size_max_internal(f, typename Analysis::Reducer{f});
  }
  template <class FunctorType, class ReducerType>
  int team_size_max(const FunctorType& f, const ReducerType& r,
                    const ParallelReduceTag&) const {
    using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                     TeamPolicyInternal, ReducerType, void>;
    return team_size_max_internal(f, typename Analysis::Reducer{r});
  }
  template <class FunctorType, class Reducer>
  int team_size_max_internal(const FunctorType& f, const Reducer& r) const {
    using FR     = CombinedFunctorReducer<FunctorType, Reducer>;
    using Driver = MetalTeamReduceDriver<FR, typename traits::work_tag>;
    // The threadgroup reduction wants a power of two.
    int t = std::min<int>(metal_max_threads<Driver>(),
                          max_threads_for(r.value_size()));
    (void)f;
    return int(Kokkos::bit_floor(unsigned(t))) / m_vector_length;
  }

  template <class FunctorType, class... Tags>
  int team_size_recommended(const FunctorType& f, const Tags&... tags) const {
    return std::max(1,
                    std::min(team_size_max(f, tags...), 256 / m_vector_length));
  }

  bool impl_auto_vector_length() const { return m_tune_vector_length; }
  bool impl_auto_team_size() const { return m_tune_team_size; }
  void impl_set_vector_length(size_t size) { m_vector_length = int(size); }
  void impl_set_team_size(size_t size) { m_team_size = int(size); }
  int impl_vector_length() const { return m_vector_length; }
  int team_size() const { return m_team_size; }
  int league_size() const { return m_league_size; }
  int chunk_size() const { return m_chunk_size; }
  execution_space space() const { return m_space; }

  size_t scratch_size(int level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] +
           size_t(team_size_) * m_thread_scratch_size[level];
  }
  size_t team_scratch_size(int level) const {
    return m_team_scratch_size[level];
  }
  size_t thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  TeamPolicyInternal& set_chunk_size(typename traits::index_type chunk) {
    m_chunk_size = int(chunk);
    return *this;
  }
  TeamPolicyInternal& set_scratch_size(int level, const PerTeamValue& team) {
    m_team_scratch_size[level] = team.value;
    return *this;
  }
  TeamPolicyInternal& set_scratch_size(int level,
                                       const PerThreadValue& thread) {
    m_thread_scratch_size[level] = thread.value;
    return *this;
  }
  TeamPolicyInternal& set_scratch_size(int level, const PerTeamValue& team,
                                       const PerThreadValue& thread) {
    m_team_scratch_size[level]   = team.value;
    m_thread_scratch_size[level] = thread.value;
    return *this;
  }

  // The device-side description of a launch with `team_size` ranks.
  template <class FunctorType>
  MetalTeamShape impl_shape(const FunctorType& f, int team_size,
                            uint32_t& groups) const {
    groups = uint32_t(std::clamp(m_league_size, 1, int(kMetalMaxTeamGroups)));
    MetalTeamShape s;
    s.league_size   = m_league_size;
    s.team_size     = team_size;
    s.vector_length = m_vector_length;
    s.scratch0      = uint32_t(
        metal_align(scratch_size(0, team_size) +
                         FunctorTeamShmemSize<FunctorType>::value(f, team_size)));
    s.scratch1_size = metal_align(scratch_size(1, team_size));
    s.scratch1 =
        s.scratch1_size
            ? static_cast<char*>(metal_team_scratch(groups * s.scratch1_size))
            : nullptr;
    if (kMetalTeamReserve + s.scratch0 > MetalRuntime::max_threadgroup_memory())
      Kokkos::abort("Kokkos::TeamPolicy<Metal>: level-0 scratch too large");
    return s;
  }
};

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::Metal> {
 public:
  using Policy = TeamPolicy<Properties...>;

 private:
  using Tag = typename Policy::work_tag;
  FunctorType m_functor;
  Policy m_policy;

 public:
  ParallelFor(const FunctorType& functor, const Policy& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    if (m_policy.league_size() == 0) return;
    using Driver = MetalTeamForDriver<FunctorType, Tag>;
    const int team_size =
        m_policy.team_size() > 0
            ? m_policy.team_size()
            : m_policy.team_size_recommended(m_functor, ParallelForTag{});
    uint32_t groups;
    MetalTeamShape shape = m_policy.impl_shape(m_functor, team_size, groups);
    const uint32_t threads =
        uint32_t(team_size * m_policy.impl_vector_length());
    if (threads > metal_max_threads<Driver>())
      Kokkos::abort("Kokkos::TeamPolicy<Metal>: team too large for the kernel");
    metal_launch(Driver{m_functor, shape}, groups, threads,
                 kMetalTeamReserve + shape.scratch0);
  }
};

template <class CombinedFunctorReducerType, class... Properties>
class ParallelReduce<CombinedFunctorReducerType,
                     Kokkos::TeamPolicy<Properties...>,
                     Kokkos::Experimental::Metal> {
 public:
  using Policy = TeamPolicy<Properties...>;

 private:
  using Tag = typename Policy::work_tag;
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
    using FR            = CombinedFunctorReducerType;
    using Driver        = MetalTeamReduceDriver<FR, Tag>;
    using Final         = MetalReduceFinal<FR>;
    const auto& f       = m_functor_reducer.get_functor();
    const auto& reducer = m_functor_reducer.get_reducer();
    const uint32_t size = reducer.value_size();
    const int team_size =
        m_policy.team_size() > 0
            ? m_policy.team_size()
            : std::max(1, std::min(m_policy.team_size_max_internal(f, reducer),
                                   256 / m_policy.impl_vector_length()));
    uint32_t groups;
    MetalTeamShape shape = m_policy.impl_shape(f, team_size, groups);
    if (m_policy.league_size() == 0) shape.league_size = 0;
    const uint32_t threads =
        uint32_t(team_size * m_policy.impl_vector_length());
    if (threads > metal_max_threads<Driver>())
      Kokkos::abort("Kokkos::TeamPolicy<Metal>: team too large for the kernel");

    char* scratch =
        static_cast<char*>(metal_scratch(metal_align(groups * size) + size));
    char* out = m_result_on_device && m_result
                    ? reinterpret_cast<char*>(m_result)
                    : scratch + metal_align(groups * size);
    metal_launch(Driver{m_functor_reducer, shape, scratch}, groups, threads,
                 kMetalTeamReserve + shape.scratch0 + threads * size);
    const uint32_t final_threads = metal_threads<Final>(0, size);
    metal_launch(Final{m_functor_reducer, scratch, groups, out}, 1,
                 final_threads, final_threads * size);
    if (m_result && !m_result_on_device) {
      Kokkos::Experimental::Metal().fence(
          "Kokkos::Metal::parallel_reduce(TeamPolicy): result to host");
      std::memcpy(m_result, out, size);
    }
  }
};

}  // namespace Kokkos::Impl

#endif
