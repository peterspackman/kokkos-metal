// TeamPolicy, nested parallelism and scratch on the Metal backend.

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "test_support.hpp"

using Exec   = Kokkos::DefaultExecutionSpace;
using Team   = Kokkos::TeamPolicy<Exec>;
using Member = Team::member_type;

// Row sums of a matrix: a team per row block, team ranks on rows, vector
// lanes on columns.
void test_nested_reduce(int team_size, int lanes) {
  std::printf("nested reduce (team %d x %d lanes)\n", team_size, lanes);
  const int rows = 1000, cols = 77;
  Kokkos::View<float**, Kokkos::LayoutRight> a("a", rows, cols);
  Kokkos::View<float*> sums("sums", rows), maxes("maxes", rows);
  Kokkos::parallel_for(
      rows, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < cols; ++j) a(i, j) = float((i * 31 + j * 7) % 13);
      });
  const int league = (rows + team_size - 1) / team_size;
  Kokkos::parallel_for(
      Team(league, team_size, lanes), KOKKOS_LAMBDA(const Member& m) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, team_size), [&](int k) {
          const int i = m.league_rank() * team_size + k;
          if (i >= rows) return;
          float s = 0, mx = 0;
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(m, cols),
              [&](int j, float& acc) { acc += a(i, j); }, s);
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(m, cols),
              [&](int j, float& acc) { acc = Kokkos::max(acc, a(i, j)); },
              Kokkos::Max<float>(mx));
          Kokkos::single(Kokkos::PerThread(m), [&] {
            sums(i)  = s;
            maxes(i) = mx;
          });
        });
      });
  auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums);
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), maxes);
  bool ok = true;
  for (int i = 0; i < rows; ++i) {
    float s = 0, mx = 0;
    for (int j = 0; j < cols; ++j) {
      s += float((i * 31 + j * 7) % 13);
      mx = std::max(mx, float((i * 31 + j * 7) % 13));
    }
    ok &= hs(i) == s && hm(i) == mx;
  }
  CHECK(ok);
}

// A team reduces a row through TeamThreadRange, and the outer reduce sums
// the teams.
void test_team_reduce(int team_size) {
  std::printf("team reduce (team %d)\n", team_size);
  const int rows = 333, cols = 1001;
  Kokkos::View<int*> out("out", rows);
  int64_t total = 0;
  Kokkos::parallel_reduce(
      Team(rows, team_size),
      KOKKOS_LAMBDA(const Member& m, int64_t& acc) {
        int s = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(m, cols),
            [&](int j, int& x) { x += (m.league_rank() + j) % 3; }, s);
        Kokkos::single(Kokkos::PerTeam(m), [&] {
          out(m.league_rank()) = s;
          acc += s;
        });
      },
      total);
  auto h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  bool ok = true;
  int64_t want_total = 0;
  for (int i = 0; i < rows; ++i) {
    int s = 0;
    for (int j = 0; j < cols; ++j) s += (i + j) % 3;
    ok &= h(i) == s;
    want_total += s;
  }
  CHECK(ok);
  CHECK(total == want_total);
}

// Level-0 scratch per team and per thread, barriers, broadcast.
void test_scratch() {
  std::puts("scratch, barriers, broadcast");
  const int league = 200, team_size = 64, n = 256;
  using ScratchF =
      Kokkos::View<float*, Exec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  Kokkos::View<float*> out("out", league), bcast("bcast", league);
  auto policy =
      Team(league, team_size)
          .set_scratch_size(0, Kokkos::PerTeam(ScratchF::shmem_size(n)),
                            Kokkos::PerThread(ScratchF::shmem_size(4)));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const Member& m) {
        ScratchF tile(m.team_scratch(0), n);
        ScratchF mine(m.thread_scratch(0), 4);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, n), [&](int i) {
          tile(i) = float(m.league_rank() + i);
        });
        mine(0) = float(m.team_rank());
        m.team_barrier();
        // Reverse through the tile: reads other ranks' writes.
        float s = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(m, n),
            [&](int i, float& x) { x += tile(n - 1 - i); }, s);
        float r = mine(0);
        m.team_broadcast(r, 5);
        Kokkos::single(Kokkos::PerTeam(m), [&] {
          out(m.league_rank())   = s;
          bcast(m.league_rank()) = r;
        });
      });
  auto h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bcast);
  bool ok = true;
  for (int l = 0; l < league; ++l) {
    float s = 0;
    for (int i = 0; i < n; ++i) s += float(l + i);
    ok &= h(l) == s && hb(l) == 5.0f;
  }
  CHECK(ok);
}

// Nested scans over threads and over vector lanes.
void test_nested_scan() {
  std::puts("nested scans");
  const int league = 50, n = 300;
  Kokkos::View<int**, Kokkos::LayoutRight> t("t", league, n), v("v", league, n);
  Kokkos::View<int*> totals("totals", league);
  Kokkos::parallel_for(
      Team(league, 32), KOKKOS_LAMBDA(const Member& m) {
        const int l = m.league_rank();
        int total   = 0;
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(m, n),
            [&](int i, int& update, bool final) {
              if (final) t(l, i) = update;
              update += (i + l) % 4;
            },
            total);
        Kokkos::single(Kokkos::PerTeam(m), [&] { totals(l) = total; });
      });
  Kokkos::parallel_for(
      Team(league, 1, 16), KOKKOS_LAMBDA(const Member& m) {
        const int l = m.league_rank();
        Kokkos::parallel_scan(Kokkos::ThreadVectorRange(m, n),
                              [&](int i, int& update, bool final) {
                                if (final) v(l, i) = update;
                                update += (i * l) % 5;
                              });
      });
  auto ht  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
  auto hv  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  auto hto = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), totals);
  bool team_ok = true, vector_ok = true, total_ok = true;
  for (int l = 0; l < league; ++l) {
    int a = 0, b = 0;
    for (int i = 0; i < n; ++i) {
      team_ok &= ht(l, i) == a;
      vector_ok &= hv(l, i) == b;
      a += (i + l) % 4;
      b += (i * l) % 5;
    }
    total_ok &= hto(l) == a;
  }
  CHECK(team_ok);
  CHECK(vector_ok);
  CHECK(total_ok);
}

void test_auto_and_large_league() {
  std::puts("AUTO team size, large league");
  const int league = 100000;
  Kokkos::View<int*> out("out", league);
  Kokkos::parallel_for(
      Team(league, Kokkos::AUTO), KOKKOS_LAMBDA(const Member& m) {
        int s = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(m, 10), [&](int j, int& x) { x += j; }, s);
        Kokkos::single(Kokkos::PerTeam(m),
                       [&] { out(m.league_rank()) = s + m.league_rank(); });
      });
  auto h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  bool ok = true;
  for (int l = 0; l < league; ++l) ok &= h(l) == 45 + l;
  CHECK(ok);
  const int max_team = Team(1, 1).team_size_max(KOKKOS_LAMBDA(const Member&){},
                                                Kokkos::ParallelForTag());
  std::printf("  team_size_max = %d\n", max_team);
  CHECK(max_team >= 256);
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  test_nested_reduce(4, 32);
  test_nested_reduce(16, 8);
  test_nested_reduce(3, 1);
  test_team_reduce(128);
  test_team_reduce(48);
  test_team_reduce(1);
  test_scratch();
  test_nested_scan();
  test_auto_and_large_league();
  std::printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
  return g_failures != 0;
}
