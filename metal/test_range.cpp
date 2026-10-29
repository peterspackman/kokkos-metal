// RangePolicy and MDRangePolicy on the Metal backend.

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <cstdint>

#include "test_support.hpp"

using Exec = Kokkos::DefaultExecutionSpace;
static_assert(std::is_same_v<Exec, Kokkos::Experimental::Metal>);

struct TagA {};
struct TagB {};

struct Tagged {
  Kokkos::View<int*> v;
  KOKKOS_FUNCTION void operator()(TagA, int i) const { v(i) = i; }
  KOKKOS_FUNCTION void operator()(TagB, int i) const { v(i) *= 3; }
};

void test_for_and_deep_copy() {
  TEST("parallel_for, deep_copy, mirrors");
  const int n = 100000;
  Kokkos::View<float*> x("x", n), y("y", n);
  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) {
        x(i) = 0.5f * i;
        y(i) = 1.0f;
      });
  Kokkos::parallel_for(
      "axpy", Kokkos::RangePolicy<>(10, n),
      KOKKOS_LAMBDA(int i) { y(i) += 2.0f * x(i); });
  auto h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  bool ok = h(0) == 1.0f && h(9) == 1.0f;
  for (int i = 10; i < n; ++i) ok &= h(i) == 1.0f + float(i);
  CHECK(ok);

  // Host -> device -> device -> host.
  Kokkos::View<double*, Kokkos::HostSpace> src("src", 1000), dst("dst", 1000);
  for (int i = 0; i < 1000; ++i) src(i) = i * 1.5;
  Kokkos::View<double*> a("a", 1000), b("b", 1000);
  Kokkos::deep_copy(a, src);
  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(dst, b);
  ok = true;
  for (int i = 0; i < 1000; ++i) ok &= dst(i) == src(i);
  CHECK(ok);

  Kokkos::deep_copy(y, 7.0f);
  Kokkos::deep_copy(h, y);
  CHECK(h(123) == 7.0f && h(n - 1) == 7.0f);

  Tagged t{Kokkos::View<int*>("t", 64)};
  Kokkos::parallel_for(Kokkos::RangePolicy<TagA>(0, 64), t);
  Kokkos::parallel_for(Kokkos::RangePolicy<TagB>(0, 64), t);
  auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t.v);
  CHECK(th(5) == 15 && th(63) == 189);
}

void test_2d_views() {
  TEST("2D views, both layouts, subviews");
  const int r = 300, c = 77;
  Kokkos::View<float**, Kokkos::LayoutRight> m("m", r, c);
  Kokkos::View<float**, Kokkos::LayoutLeft> l("l", r, c);
  Kokkos::parallel_for(
      r, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < c; ++j) {
          m(i, j) = i * 1000 + j;
          l(i, j) = m(i, j) + 1;
        }
      });
  auto row = Kokkos::subview(m, 17, Kokkos::ALL);
  Kokkos::View<float*> col("col", c);
  Kokkos::parallel_for(c, KOKKOS_LAMBDA(int j) { col(j) = row(j); });
  auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m);
  auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), l);
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), col);
  bool ok = true;
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j)
      ok &= hm(i, j) == i * 1000 + j && hl(i, j) == hm(i, j) + 1;
  for (int j = 0; j < c; ++j) ok &= hc(j) == 17000 + j;
  CHECK(ok);
}

void test_reduce() {
  TEST("parallel_reduce");
  const int64_t n = 1 << 22;
  int64_t sum     = 0;
  Kokkos::parallel_reduce(
      n, KOKKOS_LAMBDA(int64_t i, int64_t& s) { s += i % 7; }, sum);
  int64_t want = 0;
  for (int64_t i = 0; i < n; ++i) want += i % 7;
  CHECK(sum == want);

  float fsum = 0;
  Kokkos::parallel_reduce(
      1000, KOKKOS_LAMBDA(int i, float& s) { s += 0.5f; }, fsum);
  CHECK(fsum == 500.0f);

  int mx = 0, mn = 0;
  Kokkos::parallel_reduce(
      12345,
      KOKKOS_LAMBDA(int i, int& m) { m = Kokkos::max(m, (i * 7919) % 10007); },
      Kokkos::Max<int>(mx));
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(5, 12345),
      KOKKOS_LAMBDA(int i, int& m) { m = Kokkos::min(m, i); },
      Kokkos::Min<int>(mn));
  int want_mx = 0;
  for (int i = 0; i < 12345; ++i)
    want_mx = std::max(want_mx, (i * 7919) % 10007);
  CHECK(mx == want_mx);
  CHECK(mn == 5);

  // Into a device View, then read back.
  Kokkos::View<float> dv("dv");
  Kokkos::parallel_reduce(
      256, KOKKOS_LAMBDA(int i, float& s) { s += float(i); }, dv);
  float hv = 0;
  Kokkos::deep_copy(hv, dv);
  CHECK(hv == 255.0f * 128.0f);

  // Empty range: the identity.
  int empty = 42;
  Kokkos::parallel_reduce(0, KOKKOS_LAMBDA(int, int& s) { s += 1; }, empty);
  CHECK(empty == 0);

  // MinLoc.
  Kokkos::View<float*> v("v", 5000);
  Kokkos::parallel_for(
      5000, KOKKOS_LAMBDA(int i) { v(i) = Kokkos::fabs(float(i) - 3210.25f); });
  Kokkos::MinLoc<float, int>::value_type loc;
  Kokkos::parallel_reduce(
      5000,
      KOKKOS_LAMBDA(int i, Kokkos::MinLoc<float, int>::value_type& l) {
        if (v(i) < l.val) {
          l.val = v(i);
          l.loc = i;
        }
      },
      Kokkos::MinLoc<float, int>(loc));
  CHECK(loc.loc == 3210);
}

void test_scan() {
  TEST("parallel_scan");
  for (int n : {1, 7, 256, 1000, 100003}) {
    Kokkos::View<int*> in("in", n), out("out", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) { in(i) = i % 5 + 1; });
    int total = 0;
    Kokkos::parallel_scan(
        n,
        KOKKOS_LAMBDA(int i, int& update, bool final) {
          if (final) out(i) = update;
          update += in(i);
        },
        total);
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    int run = 0;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      ok &= ho(i) == run;
      run += i % 5 + 1;
    }
    CHECK(ok);
    CHECK(total == run);
  }
}

void test_mdrange() {
  TEST("MDRangePolicy");
  const int a = 37, b = 41, c = 5;
  Kokkos::View<int***> v("v", a, b, c);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {a, b, c}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        v(i, j, k) = i * 10000 + j * 10 + k;
      });
  auto h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  bool ok = true;
  for (int i = 0; i < a; ++i)
    for (int j = 0; j < b; ++j)
      for (int k = 0; k < c; ++k) ok &= h(i, j, k) == i * 10000 + j * 10 + k;
  CHECK(ok);

  long sum = 0;
  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2, 3}, {a, b}),
      KOKKOS_LAMBDA(int i, int j, long& s) { s += i * j; }, sum);
  long want = 0;
  for (int i = 2; i < a; ++i)
    for (int j = 3; j < b; ++j) want += i * j;
  CHECK(sum == want);
}

void test_math_and_atomics() {
  TEST("math and atomics");
  const int n = 4096;
  Kokkos::View<float*> r("r", n);
  Kokkos::View<int*> hist("hist", 8);
  Kokkos::View<float> acc("acc");
  Kokkos::parallel_for(
      n, KOKKOS_LAMBDA(int i) {
        float x = 0.001f * i;
        r(i)    = Kokkos::exp(-x) * Kokkos::cos(x) + Kokkos::sqrt(x) +
               Kokkos::tanh(x) + Kokkos::pow(x, 1.5f);
        Kokkos::atomic_inc(&hist(i % 8));
        Kokkos::atomic_add(&acc(), 0.5f);
      });
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hist);
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc);
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float x    = 0.001f * i;
    float want = std::exp(-x) * std::cos(x) + std::sqrt(x) + std::tanh(x) +
                 std::pow(x, 1.5f);
    ok &= std::fabs(hr(i) - want) <= 2e-6f * (1 + std::fabs(want));
  }
  CHECK(ok);
  for (int k = 0; k < 8; ++k) CHECK(hh(k) == n / 8);
  CHECK(ha() == 0.5f * n);
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Exec().print_configuration(std::cout);
  test_for_and_deep_copy();
  test_2d_views();
  test_reduce();
  test_scan();
  test_mdrange();
  test_math_and_atomics();
  std::printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
  return g_failures != 0;
}
