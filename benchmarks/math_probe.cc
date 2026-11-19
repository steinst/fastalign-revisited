// Math probe: evaluates fast_align's own numerical functions (included from the reference
// source, not copied) on many inputs and prints inputs and results as raw IEEE-754 bits, the
// test vectors for crates/fastalign-core/src/math.rs. Built with the reference build's flags by
// benchmarks/math_vectors.sh. Inputs come from a fixed-seed generator at run time, so the
// compiler cannot constant-fold any libm call.
//
// Output lines (doubles as 16 hex digits of their bits, integers in decimal):
//   feature I J M N RESULT
//   uprob I J M N LAMBDA RESULT
//   z I M N LAMBDA RESULT
//   dlogz I M N LAMBDA RESULT
//   digamma X RESULT
//   logpois X LAMBDA RESULT
//   pow2 X RESULT                 pow(2.0, X) as in the perplexity statistic
//   g X TEXT                      `std::cout << X` (the C++ stream default format)
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

#include "src/da.h"
#include "src/ttables.h"

static uint64_t bits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, sizeof u);
  return u;
}
static double from_bits(uint64_t u) {
  double x;
  std::memcpy(&x, &u, sizeof x);
  return x;
}

int main() {
  std::mt19937_64 rng(20261118);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  auto lambda = [&]() {
    switch (rng() % 4) {
      case 0: return 4.0;
      case 1: return 0.1 + unit(rng) * 13.9;
      case 2: return unit(rng) * 30.0;
      default: return 0.1 + unit(rng);
    }
  };
  for (int k = 0; k < 3000; ++k) {
    unsigned m = 1 + rng() % 120, n = 1 + rng() % 120;
    unsigned i = 1 + rng() % (m + 3), j = rng() % (n + 2);
    double l = lambda();
    printf("feature %u %u %u %u %016" PRIx64 "\n", i, j, m, n,
           bits(DiagonalAlignment::Feature(i, j, m, n)));
    printf("uprob %u %u %u %u %016" PRIx64 " %016" PRIx64 "\n", i, j, m, n, bits(l),
           bits(DiagonalAlignment::UnnormalizedProb(i, j, m, n, l)));
  }
  for (int k = 0; k < 6000; ++k) {
    unsigned m = 1 + rng() % (k < 3000 ? 60 : 400), n = 1 + rng() % (k < 3000 ? 60 : 400);
    unsigned i = 1 + rng() % m;
    double l = lambda();
    printf("z %u %u %u %016" PRIx64 " %016" PRIx64 "\n", i, m, n, bits(l),
           bits(DiagonalAlignment::ComputeZ(i, m, n, l)));
    printf("dlogz %u %u %u %016" PRIx64 " %016" PRIx64 "\n", i, m, n, bits(l),
           bits(DiagonalAlignment::ComputeDLogZ(i, m, n, l)));
  }
  for (int k = 0; k < 10000; ++k) {
    double x;
    switch (k % 5) {
      case 0: x = unit(rng) * 10.0; break;
      case 1: x = 0.01 + unit(rng) * 3.0; break;            // count + alpha
      case 2: x = 0.01 + (rng() % 1000) * 1e-6; break;      // tiny counts
      case 3: x = unit(rng) * 1e6; break;
      default: x = from_bits(0x3e00000000000000ULL + (rng() >> 8)); break;  // tiny..huge
    }
    printf("digamma %016" PRIx64 " %016" PRIx64 "\n", bits(x), bits(Md::digamma(x)));
  }
  for (int k = 0; k < 2000; ++k) {
    unsigned x = rng() % 400;
    double l = 0.05 + unit(rng) * (k % 2 ? 400.0 : 40.0);
    printf("logpois %u %016" PRIx64 " %016" PRIx64 "\n", x, bits(l),
           bits(Md::log_poisson(x, l)));
  }
  for (int k = 0; k < 2000; ++k) {
    double x = unit(rng) * (k % 2 ? 40.0 : 12.0);
    printf("pow2 %016" PRIx64 " %016" PRIx64 "\n", bits(x), bits(pow(2.0, x)));
  }
  for (int k = 0; k < 12000; ++k) {
    double x;
    switch (k % 4) {
      case 0: x = from_bits(rng()); break;
      case 1: x = (unit(rng) - 0.5) * pow(10.0, double(int(rng() % 40) - 20)); break;
      case 2: x = double(int64_t(rng() % 20000000) - 10000000) / 2.0; break;
      default: x = -std::log(unit(rng)) * -10.0; break;
    }
    std::ostringstream s;
    s << x;
    printf("g %016" PRIx64 " %s\n", bits(x), s.str().c_str());
  }
  return 0;
}
