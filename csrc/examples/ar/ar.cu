//
// Created by Osayamen on 3/30/26.
//
// non-atomic allredu
#include <cstdio>
#include <cutlass/array.h>

template<int a>
  struct foo {
  static void work(const int& x, const int* __restrict__ const& y) {
    printf("inside foo<a>: %d, %p\n", x, y);
  }
};

template<>
struct foo<1> {
  static void work(const int& x, const int* __restrict__ const&) {
    printf("inside foo<1>, x: %d\n", x);
  }
};

int main() {
  constexpr std::array<int, 2> a{};
  foo<0>::work(4, a.data());
  foo<1>::work(4, nullptr);
}