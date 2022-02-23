/// @file approx.hpp
#pragma once
#include <complex>       // abs
#include <kspc/core.hpp> // cast_if_needed

namespace kspc::approx {
  /// @addtogroup approx
  /// @{

  /// @cond
  namespace detail {
    template <typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> //
    less(const T& t1, const T& t2, const T& eps) noexcept {
      // avoid the subtraction with infinity
      return t1 + eps < t2;
    }

    template <typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> //
    greater(const T& t1, const T& t2, const T& eps) noexcept {
      // avoid the subtraction with infinity
      return t2 + eps < t1;
    }

    template <typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> //
    not_equal_to(const T& t1, const T& t2, const T& eps) noexcept {
      return less(t1, t2, eps) || greater(t1, t2, eps);
    }
  } // namespace detail
  /// @endcond

  template <typename T1, typename T2, typename U>
  constexpr std::enable_if_t<
    std::conjunction_v<std::is_arithmetic<T1>, std::is_arithmetic<T2>, std::is_arithmetic<U>>, bool>
  less(const T1& t1, const T2& t2, const U& eps) noexcept {
    using V = std::common_type_t<T1, T2>;
    return detail::less(cast_if_needed<V>(t1), cast_if_needed<V>(t2), cast_if_needed<V>(eps));
  }

  template <typename T1, typename T2, typename U>
  constexpr std::enable_if_t<
    std::conjunction_v<std::is_arithmetic<T1>, std::is_arithmetic<T2>, std::is_arithmetic<U>>, bool>
  greater(const T1& t1, const T2& t2, const U& eps) noexcept {
    using V = std::common_type_t<T1, T2>;
    return detail::greater(cast_if_needed<V>(t1), cast_if_needed<V>(t2), cast_if_needed<V>(eps));
  }

  template <typename T1, typename T2, typename U>
  constexpr auto less_equal(const T1& t1, const T2& t2, const U& eps) noexcept
    -> decltype(!greater(t1, t2, eps)) {
    return !greater(t1, t2, eps);
  }

  template <typename T1, typename T2, typename U>
  constexpr auto greater_equal(const T1& t1, const T2& t2, const U& eps) noexcept
    -> decltype(!less(t1, t2, eps)) {
    return !less(t1, t2, eps);
  }

  template <typename T1, typename T2, typename U>
  constexpr std::enable_if_t<
    std::conjunction_v<std::is_arithmetic<T1>, std::is_arithmetic<T2>, std::is_arithmetic<U>>, bool>
  not_equal_to(const T1& t1, const T2& t2, const U& eps) noexcept {
    using V = std::common_type_t<T1, T2>;
    return detail::not_equal_to(cast_if_needed<V>(t1), cast_if_needed<V>(t2),
                                cast_if_needed<V>(eps));
  }

  template <typename T, typename U>
  constexpr std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>, std::is_arithmetic<U>>, bool>
  not_equal_to(const std::complex<T>& t1, const std::complex<T>& t2, const U& eps) noexcept {
    return std::abs(t1 - t2) > T(eps);
  }

  template <typename T, typename U>
  constexpr std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>, std::is_arithmetic<U>>, bool>
  not_equal_to(const std::complex<T>& t1, const T& t2, const U& eps) noexcept {
    return std::abs(t1 - t2) > T(eps);
  }

  template <typename T, typename U>
  constexpr std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>, std::is_arithmetic<U>>, bool>
  not_equal_to(const T& t1, const std::complex<T>& t2, const U& eps) noexcept {
    return std::abs(t1 - t2) > T(eps);
  }

  template <typename T1, typename T2, typename U>
  constexpr auto equal_to(const T1& t1, const T2& t2, const U& eps) noexcept
    -> decltype(!not_equal_to(t1, t2, eps)) {
    return !not_equal_to(t1, t2, eps);
  }

  /// @}
} // namespace kspc::approx
