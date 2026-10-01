#ifndef PURLIN_STATIC_FOR_CUH
#define PURLIN_STATIC_FOR_CUH

#include <cuda/std/type_traits>

namespace purlin {
  namespace detail {
    template<typename Function, typename Index, Index... Indices>
    __device__ __forceinline__
    constexpr void static_for_impl(Function& function, cuda::std::integer_sequence<Index, Indices...>)
      noexcept((noexcept(function(cuda::std::integral_constant<Index, Indices>{})) && ...)) {
      (static_cast<void>(function(cuda::std::integral_constant<Index, Indices>{})), ...);
    }
  }

  template<auto Count, typename Function>
  __device__ __forceinline__
  constexpr void static_for(Function&& function)
    noexcept(noexcept(detail::static_for_impl(function,
      cuda::std::make_integer_sequence<decltype(Count), Count>{}))) {
    static_assert(Count >= 0, "static_for requires a non-negative count");
    detail::static_for_impl(function, cuda::std::make_integer_sequence<decltype(Count), Count>{});
  }
}

#endif // PURLIN_STATIC_FOR_CUH
