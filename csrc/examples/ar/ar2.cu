//
// Created by Osayamen on 3/30/26.
//
// non-atomic allredu
#include <cstdio>
#include <cutlass/array.h>
int main() {
  cutlass::AlignedArray<float, 1> a{};
  std::array<float, a.size()> b{};
  printf("%f\n", b[0]);
}