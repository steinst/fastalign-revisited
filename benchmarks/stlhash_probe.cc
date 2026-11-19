// stlhash_probe: prints libstdc++ std::unordered_map iteration order for given insertion
// sequences. Test oracle for crates/fastalign-core/src/stlhash.rs (see docs/SPEC-fast-align.md,
// "libstdc++ unordered_map iteration order").
//
// Build: g++ -std=c++11 -O2 -o stlhash_probe benchmarks/stlhash_probe.cc
// Input (stdin), one case per line:
//   u32  NAME k1 k2 ...          keys for unordered_map<unsigned,double> (inserted with m[k] = 0)
//   pair NAME a,b a,b ...        keys for fast_align's size_counts map (++m[make_pair(a,b)])
// Output per case:
//   case NAME KIND N             N = number of insert calls (duplicates included)
//   bkt  b1 b2 ...               bucket_count() after each insert call
//   order k ...                  iteration order after all inserts
//   copy k ...                   order of a default-constructed map after `copy = orig`
//   swap k ...                   order of vector element 0 after vector<map>::swap (as Normalize)
// `./stlhash_probe primes` prints libstdc++'s __prime_list (bucket-count table) instead.
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace std { namespace __detail { extern const unsigned long __prime_list[]; } }

struct PairHash {  // same hash as fast_align's size_counts
  size_t operator()(const std::pair<short, short>& x) const {
    return (unsigned short)x.first << 16 | (unsigned)x.second;
  }
};

template <class Map, class Key, class Print>
void run(const std::string& name, const char* kind, const std::vector<Key>& keys, Print print) {
  std::vector<Map> a(1), b(1);
  std::cout << "case " << name << ' ' << kind << ' ' << keys.size() << "\nbkt";
  for (const Key& k : keys) { ++a[0][k]; std::cout << ' ' << a[0].bucket_count(); }
  std::cout << "\norder";
  for (const auto& kv : a[0]) print(kv.first);
  b[0] = a[0];
  std::cout << "\ncopy";
  for (const auto& kv : b[0]) print(kv.first);
  a.swap(b);
  std::cout << "\nswap";
  for (const auto& kv : b[0]) print(kv.first);
  std::cout << '\n';
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "primes") {
    // 256 + 48 + 1 entries on LP64 (symbol size 0x988 bytes); the last is a sentinel.
    for (int i = 0; i < 305; ++i) std::printf("%lu\n", std::__detail::__prime_list[i]);
    return 0;
  }
  std::string line, kind, name;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    if (!(in >> kind >> name)) continue;
    if (kind == "u32") {
      std::vector<unsigned> keys;
      for (unsigned k; in >> k;) keys.push_back(k);
      run<std::unordered_map<unsigned, double>>(name, "u32", keys,
                                                [](unsigned k) { std::cout << ' ' << k; });
    } else if (kind == "pair") {
      std::vector<std::pair<short, short>> keys;
      for (std::string t; in >> t;) {
        const size_t c = t.find(',');
        keys.push_back(std::make_pair<short, short>(std::stoi(t.substr(0, c)),
                                                    std::stoi(t.substr(c + 1))));
      }
      run<std::unordered_map<std::pair<short, short>, unsigned, PairHash>>(
          name, "pair", keys, [](const std::pair<short, short>& k) {
            std::cout << ' ' << k.first << ',' << k.second;
          });
    }
  }
  return 0;
}
