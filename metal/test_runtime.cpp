// The plugin and runtime without Kokkos: kernels written against the
// intrinsics directly, checked against the same code run on the host.

#include <Metal/Kokkos_Metal_Intrinsics.hpp>
#include <Metal/Kokkos_Metal_Runtime.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rt = Kokkos::Impl::MetalRuntime;

namespace {

int failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

template <class F>
struct Range {
  F f;
  int64_t n;
  void metal_run() const {
    int64_t i = __kmetal_thread_position_in_grid(0);
    if (i < n) f(i);
  }
};

template <class F>
void parallel_for(int64_t n, const F& f) {
  Range<F> d{f, n};
  const void* key = Kokkos::Impl::MetalDevice::Kernel<Range<F>>::instantiate();
  uint32_t groups[3]  = {uint32_t((n + 255) / 256), 1, 1},
           threads[3] = {256, 1, 1};
  rt::dispatch(key, &d, sizeof d, groups, threads, 0);
}

// One threadgroup per block of 256, summing it through threadgroup memory.
template <class F>
struct BlockSum {
  F f;
  int64_t n;
  float* out;
  void metal_run() const {
    auto* s    = static_cast<float*>(__kmetal_threadgroup_memory());
    uint32_t t = __kmetal_thread_position_in_threadgroup(0);
    int64_t i  = __kmetal_thread_position_in_grid(0);
    s[t]       = i < n ? f(i) : 0.0f;
    for (uint32_t w = 128; w > 0; w /= 2) {
      __kmetal_threadgroup_barrier();
      if (t < w) s[t] += s[t + w];
    }
    if (t == 0) out[__kmetal_threadgroup_position_in_grid(0)] = s[0];
  }
};

template <class T>
T* alloc(size_t n) {
  return static_cast<T*>(rt::allocate(n * sizeof(T)));
}

int collatz_steps(int64_t x) {
  int steps = 0;
  while (x != 1) {
    x = (x % 2) ? 3 * x + 1 : x / 2;
    if (++steps > 1000) break;
  }
  return steps;
}

// Nested loops with break and continue, a local array indexed at run time.
float tangle(int64_t i) {
  float local[8];
  for (int k = 0; k < 8; ++k) local[k] = float((i * 7 + k) % 5);
  float acc = 0;
  for (int a = 0; a < 6; ++a) {
    if (a == int(i % 6)) continue;
    for (int b = 0; b < 8; ++b) {
      if (local[(a + b + i) % 8] > 3) break;
      acc += local[(a * b) % 8] * 0.5f;
      if (acc > 40) goto done;
    }
  }
done:
  return acc + std::sqrt(float(i));
}

void test_axpy() {
  std::puts("axpy");
  const int64_t n = 100003;
  float *x = alloc<float>(n), *y = alloc<float>(n);
  for (int64_t i = 0; i < n; ++i) x[i] = float(i), y[i] = 1.0f;
  const float a = 2.5f;
  parallel_for(n, [=](int64_t i) { y[i] = a * x[i] + y[i]; });
  rt::fence();
  bool ok = true;
  for (int64_t i = 0; i < n; ++i) ok &= y[i] == 2.5f * float(i) + 1.0f;
  CHECK(ok);
  rt::deallocate(x);
  rt::deallocate(y);
}

void test_control_flow() {
  std::puts("control flow");
  const int64_t n = 5000;
  int* steps      = alloc<int>(n);
  float* t        = alloc<float>(n);
  parallel_for(n, [=](int64_t i) {
    steps[i] = collatz_steps(i + 1);
    t[i]     = tangle(i);
  });
  rt::fence();
  int bad = 0;
  for (int64_t i = 0; i < n; ++i) {
    bad += steps[i] != collatz_steps(i + 1);
    bad += std::fabs(t[i] - tangle(i)) > 1e-5f * (1 + std::fabs(tangle(i)));
  }
  CHECK(bad == 0);
  rt::deallocate(steps);
  rt::deallocate(t);
}

void test_atomics() {
  std::puts("atomics");
  const int64_t n = 1 << 20;
  int* hist       = alloc<int>(16);
  float* total    = alloc<float>(1);
  unsigned* top   = alloc<unsigned>(1);
  for (int k = 0; k < 16; ++k) hist[k] = 0;
  *total = 0;
  *top   = 0;
  parallel_for(n, [=](int64_t i) {
    __atomic_fetch_add(&hist[(i * 2654435761u) % 16], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(total, 0.25f, __ATOMIC_RELAXED);
    __atomic_fetch_max(top, unsigned(i % 1000), __ATOMIC_RELAXED);
  });
  rt::fence();
  int64_t sum = 0;
  for (int k = 0; k < 16; ++k) sum += hist[k];
  CHECK(sum == n);
  CHECK(*total == 0.25f * n);
  CHECK(*top == 999);
  rt::deallocate(hist);
  rt::deallocate(total);
  rt::deallocate(top);
}

void test_threadgroup_memory() {
  std::puts("threadgroup memory");
  const int64_t n      = 256 * 37 + 11;
  const int64_t groups = (n + 255) / 256;
  float* out           = alloc<float>(groups);
  auto f               = [](int64_t i) { return float(i % 3); };
  BlockSum<decltype(f)> d{f, n, out};
  const void* key =
      Kokkos::Impl::MetalDevice::Kernel<BlockSum<decltype(f)>>::instantiate();
  uint32_t g[3] = {uint32_t(groups), 1, 1}, t[3] = {256, 1, 1};
  rt::dispatch(key, &d, sizeof d, g, t, 256 * sizeof(float));
  rt::fence();
  double sum = 0, want = 0;
  for (int64_t b = 0; b < groups; ++b) sum += out[b];
  for (int64_t i = 0; i < n; ++i) want += f(i);
  CHECK(sum == want);
  rt::deallocate(out);
}

// A functor with an array member and a pointer to a constant table.
struct Poly {
  std::array<float, 4> c;
  const float* x;
  float* y;
  void operator()(int64_t i) const {
    static constexpr float scale[3] = {1.0f, -1.0f, 0.5f};
    float v = x[i], r = 0;
    for (int k = 3; k >= 0; --k) r = r * v + c[k];
    y[i] = r * scale[i % 3];
  }
};

void test_functor() {
  std::puts("functor");
  const int64_t n = 1000;
  float *x = alloc<float>(n), *y = alloc<float>(n);
  for (int64_t i = 0; i < n; ++i) x[i] = 0.001f * i;
  Poly p{{1, 2, 3, 4}, x, y};
  parallel_for(n, p);
  rt::fence();
  std::vector<float> want(n);
  float* keep = p.y;
  p.y         = want.data();
  for (int64_t i = 0; i < n; ++i) p(i);
  int bad = 0;
  for (int64_t i = 0; i < n; ++i) bad += std::fabs(keep[i] - want[i]) > 1e-5f;
  CHECK(bad == 0);
  rt::deallocate(x);
  rt::deallocate(y);
}

// Software fp64 against the CPU, bit for bit: random patterns, with
// specials and subnormals mixed in.
struct Fp64Out {
  double add, sub, mul, div, sqrt, floor, round, rint;
  float to_float;
  int64_t to_int;
  double from_int, from_float;
  uint32_t cmp;
};

uint64_t mix(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

double pick(uint64_t& s) {
  static const double special[] = {
      0.0,      -0.0,      1.0,       -1.0,  0.5,
      2.0,      1e-310,    -4.9e-324, 1e308, -1.7976931348623157e308,
      INFINITY, -INFINITY, NAN,       3.0,   1.0 / 3.0};
  uint64_t r = mix(s);
  if (r % 8 == 0)
    return special[(r >> 8) % (sizeof special / sizeof special[0])];
  if (r % 8 == 1) {  // near another value: cancellation
    uint64_t b = mix(s) & 0x3fffffffffffffffull;
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  }
  double d;
  std::memcpy(&d, &r, 8);
  return std::isnan(d) ? 1.5 : d;
}

Fp64Out fp64_ops(double a, double b) {
  Fp64Out o;
  o.add      = a + b;
  o.sub      = a - b;
  o.mul      = a * b;
  o.div      = a / b;
  o.sqrt     = std::sqrt(std::fabs(a));
  o.floor    = std::floor(a);
  o.round    = std::round(a);
  o.rint     = std::rint(b);
  o.to_float = float(a);
  o.to_int   = std::fabs(a) < 9e18 ? int64_t(a) : 0;
  o.from_int = double(
      int64_t(mix(*reinterpret_cast<uint64_t*>(&b)) >> (uint64_t(a) & 31)));
  o.from_float = double(float(b));
  o.cmp =
      (a < b) | (a <= b) << 1 | (a == b) << 2 | (a > b) << 3 | (a != b) << 4;
  return o;
}

bool same(double x, double y) {
  if (std::isnan(x) && std::isnan(y)) return true;
  uint64_t a, b;
  std::memcpy(&a, &x, 8);
  std::memcpy(&b, &y, 8);
  return a == b;
}

void test_fp64() {
  std::puts("fp64 emulation");
  const int n = 1 << 18;
  double *a = alloc<double>(n), *b = alloc<double>(n);
  Fp64Out* out  = alloc<Fp64Out>(n);
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < n; ++i) a[i] = pick(seed), b[i] = pick(seed);
  parallel_for(n, [=](int64_t i) { out[i] = fp64_ops(a[i], b[i]); });
  rt::fence();
  int bad = 0;
  for (int i = 0; i < n && bad < 10; ++i) {
    Fp64Out w = fp64_ops(a[i], b[i]), g = out[i];
    const char* what = !same(w.add, g.add)       ? "add"
                       : !same(w.sub, g.sub)     ? "sub"
                       : !same(w.mul, g.mul)     ? "mul"
                       : !same(w.div, g.div)     ? "div"
                       : !same(w.sqrt, g.sqrt)   ? "sqrt"
                       : !same(w.floor, g.floor) ? "floor"
                       : !same(w.round, g.round) ? "round"
                       : !same(w.rint, g.rint)   ? "rint"
                       : !(w.to_float == g.to_float ||
                           (std::isnan(w.to_float) && std::isnan(g.to_float)))
                           ? "to_float"
                       : w.to_int != g.to_int              ? "to_int"
                       : !same(w.from_int, g.from_int)     ? "from_int"
                       : !same(w.from_float, g.from_float) ? "from_float"
                       : w.cmp != g.cmp                    ? "cmp"
                                                           : nullptr;
    if (what) {
      std::printf("  %s(%a, %a) differs\n", what, a[i], b[i]);
      ++bad;
    }
  }
  CHECK(bad == 0);
  rt::deallocate(a);
  rt::deallocate(b);
  rt::deallocate(out);
}

// The built-in GEMM against a plain loop: every transpose, ragged sizes,
// beta and bias; then its speed on a shape PET runs.
void test_gemm() {
  std::puts("gemm");
  struct Case {
    int m, n, k;
  };
  for (Case c : {Case{1, 1, 1}, Case{70, 33, 45}, Case{128, 64, 32},
                 Case{200, 130, 77}}) {
    for (float beta : {0.5f, 0.0f})
      for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb) {
          const int lda = ta ? c.m : c.k, ldb = tb ? c.k : c.n, ldc = c.n;
          // 32 floats in, as a View's data is after its allocation's header.
          float *a      = alloc<float>(c.m * c.k + 32) + 32,
                *b      = alloc<float>(c.k * c.n + 32) + 32,
                *out    = alloc<float>(c.m * c.n + 32) + 32,
                *bias   = alloc<float>(c.n + 32) + 32;
          uint64_t seed = 12345;
          for (int i = 0; i < c.m * c.k; ++i)
            a[i] = float(mix(seed) % 1000) / 500 - 1;
          for (int i = 0; i < c.k * c.n; ++i)
            b[i] = float(mix(seed) % 1000) / 500 - 1;
          for (int i = 0; i < c.m * c.n; ++i) out[i] = float(i % 7);
          for (int j = 0; j < c.n; ++j) bias[j] = float(j) * 0.25f;
          std::vector<float> want(out, out + c.m * c.n);
          for (int i = 0; i < c.m; ++i)
            for (int j = 0; j < c.n; ++j) {
              double acc = 0;
              for (int p = 0; p < c.k; ++p)
                acc += double(ta ? a[p * lda + i] : a[i * lda + p]) *
                       (tb ? b[j * ldb + p] : b[p * ldb + j]);
              want[i * ldc + j] =
                  float(1.5 * acc + beta * want[i * ldc + j] + bias[j]);
            }
          rt::gemm(ta, tb, c.m, c.n, c.k, 1.5f, a, lda, b, ldb, beta, out, ldc,
                   bias);
          rt::fence();
          double err = 0;
          for (int i = 0; i < c.m * c.n; ++i)
            err = std::max(err, double(std::fabs(out[i] - want[i])));
          if (err > 1e-3)
            std::printf("  %dx%dx%d ta=%d tb=%d beta=%g: max error %g\n", c.m,
                        c.n, c.k, ta, tb, beta, err);
          CHECK(err <= 1e-3);
          rt::deallocate(a - 32), rt::deallocate(b - 32),
              rt::deallocate(out - 32), rt::deallocate(bias - 32);
        }
  }
  const int m = 48350, n = 256, k = 128;
  float *a = alloc<float>(m * k), *b = alloc<float>(k * n),
        *c = alloc<float>(m * n);
  for (int tb = 0; tb < 2; ++tb) {
    rt::gemm(false, tb, m, n, k, 1, a, k, b, tb ? k : n, 0, c, n);
    rt::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 20; ++r)
      rt::gemm(false, tb, m, n, k, 1, a, k, b, tb ? k : n, 0, c, n);
    rt::fence();
    double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() /
        20;
    std::printf("  %dx%dx%d %s: %.3f ms, %.2f TFLOPS\n", m, n, k,
                tb ? "NT" : "NN", s * 1e3, 2.0 * m * n * k / s / 1e12);
  }
  rt::deallocate(a), rt::deallocate(b), rt::deallocate(c);
}

}  // namespace

int main(int argc, char** argv) {
  // An argument runs only the test of that name.
  auto want = [&](const char* name) {
    return argc < 2 || std::strcmp(name, argv[1]) == 0;
  };
  rt::initialize();
  std::printf("device: %s\n", rt::device_name());
  if (want("test_axpy")) test_axpy();
  if (want("test_control_flow")) test_control_flow();
  if (want("test_atomics")) test_atomics();
  if (want("test_threadgroup_memory")) test_threadgroup_memory();
  if (want("test_functor")) test_functor();
  if (want("test_fp64")) test_fp64();
  if (want("test_gemm")) test_gemm();
  rt::finalize();
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures != 0;
}
