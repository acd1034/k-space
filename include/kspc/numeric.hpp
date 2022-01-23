/// @file numeric.hpp
#pragma once
#include <algorithm> // min
#include <array>
#include <functional> // invoke
#include <vector>
#include <kspc/core.hpp> // is_range, identity_fn, conj_fn

// clang-format off

// Numerical algorithms
namespace kspc {
  /// @addtogroup numeric
  /// @{

  // sum

  /// @cond
  namespace detail {
    // NOTE: irrelevant to `std::projected`
    template <typename P, typename I>
    using projected_t = std::invoke_result_t<P&, iter_reference_t<I>>;

    template <typename I, typename T,
              typename BOp = std::plus<>,
              typename P = identity_fn>
    struct sum_constraints : std::conjunction<
      std::is_invocable<P&, iter_reference_t<I>>,
      std::is_invocable<BOp&, T, projected_t<P, I>>> {};

    template <typename I, typename S, typename T,
              typename BOp = std::plus<>,
              typename P = identity_fn,
              std::enable_if_t<std::conjunction_v<
                is_sentinel_for<S, I>,
                is_input_iterator<I>,
                sum_constraints<I, T, BOp, P>>, std::nullptr_t> = nullptr>
    constexpr auto sum(I first, S last, T init, BOp bop = {}, P proj = {}) {
      using U = remove_cvref_t<std::invoke_result_t<BOp&, T, projected_t<P, I>>>;
      U ret = std::move(init);
      for (; first != last; ++first) {
        ret = std::invoke(bop, std::move(ret), std::invoke(proj, *first));
      }
      return ret;
    }
  } // namespace detail
  /// @endcond

  /// @brief `std::accumulate` without the initial value
  /// @note The order of the arguments `P`, `BOp` is different from range-v3.
  template <typename R,
            typename P = identity_fn,
            typename BOp = std::plus<>,
            std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
  constexpr auto sum(R&& r, P proj = {}, BOp bop = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r));

    auto first = begin(r);
    auto init = std::invoke(proj, *first++);
    return detail::sum(first, end(r), init, std::move(bop), std::move(proj));
  }

  // innerp

  /// @cond
  namespace detail {
    template <typename I1, typename I2, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity_fn>
    struct innerp_constraints : std::conjunction<
      std::is_invocable<P1&, iter_value_t<I1>>,
      std::is_invocable<P2&, iter_value_t<I2>>,
      std::is_invocable<BOp2&, projected_t<P1, I1>, projected_t<P2, I2>>,
      std::is_invocable<
        BOp1&,
        T,
        std::invoke_result_t<BOp2&, projected_t<P1, I1>, projected_t<P2, I2>>>> {};

    template <typename I1, typename S1, typename I2, typename S2, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity_fn,
              std::enable_if_t<std::conjunction_v<
                is_sentinel_for<S1, I1>,
                is_sentinel_for<S2, I2>,
                is_input_iterator<I1>,
                is_input_iterator<I2>,
                innerp_constraints<I1, I2, T, BOp1, BOp2, P1, P2>>, std::nullptr_t> = nullptr>
    constexpr auto innerp(I1 first1, S1 last1, I2 first2, S2 last2, T init,
                          BOp1 bop1 = {}, BOp2 bop2 = {}, P1 proj1 = {}, P2 proj2 = {}) {
      using U = remove_cvref_t<
        std::invoke_result_t<
          BOp1&,
          T,
          std::invoke_result_t<BOp2&, projected_t<P1, I1>, projected_t<P2, I2>>>>;
      U ret = std::move(init);
      for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
        ret =
          std::invoke(
            bop1,
            std::move(ret),
            std::invoke(bop2, std::invoke(proj1, *first1), std::invoke(proj2, *first2)));
      }
      return ret;
    }
  } // namespace detail
  /// @endcond

  /// @brief `std::inner_product` without the initial value
  /// @note The order of the arguments `P#`, `BOp#` is different from range-v3.
  template <typename R1, typename R2,
            typename P1 = conj_fn,
            typename P2 = identity_fn,
            typename BOp1 = std::plus<>,
            typename BOp2 = std::multiplies<>,
            std::enable_if_t<std::conjunction_v<
              is_range<R1>,
              is_range<R2>,
              std::negation<is_range<P1>>>, std::nullptr_t> = nullptr>
  constexpr auto innerp(R1&& r1, R2&& r2,
                        P1 proj1 = {}, P2 proj2 = {}, BOp1 bop1 = {}, BOp2 bop2 = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r1));
    assert(!empty(r2));

    auto first1 = begin(r1);
    auto first2 = begin(r2);
    auto init = std::invoke(
                  bop2,
                  std::invoke(proj1, *first1++),
                  std::invoke(proj2, *first2++));
    return detail::innerp(
             first1, end(r1), first2, end(r2), init,
             std::move(bop1),
             std::move(bop2),
             std::move(proj1),
             std::move(proj2));
  }

  /// @}
} // namespace kspc

// clang-format on

// arithmetic operators for `std::array` and `std::vector`
namespace kspc {
  /// @addtogroup numeric
  /// @{

  inline namespace arithmetic_ops {
    // std::array

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto& operator+(const std::array<T, N>& x) {
      return x;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::array<T, N>& x) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = -x[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(std::array<T, N>&& x) {
      for (std::size_t i = 0; i < N; ++i) x[i] = -x[i];
      return std::move(x);
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator+(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] + y[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] - y[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const T& val, const std::array<T, N>& x) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = val * x[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const T& val, std::array<T, N>&& x) {
      for (std::size_t i = 0; i < N; ++i) x[i] = val * x[i];
      return std::move(x);
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const std::array<T, N>& x, const T& val) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] * val;
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(std::array<T, N>&& x, const T& val) {
      for (std::size_t i = 0; i < N; ++i) x[i] = x[i] * val;
      return std::move(x);
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(const std::array<T, N>& x, const T& val) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] / val;
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(std::array<T, N>&& x, const T& val) {
      for (std::size_t i = 0; i < N; ++i) x[i] = x[i] / val;
      return std::move(x);
    }

    // std::vector

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto& operator+(const std::vector<T>& x) {
      return x;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::vector<T>& x) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = -x[i];
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(std::vector<T>&& x) {
      const auto n = std::size(x);
      for (std::size_t i = 0; i < n; ++i) x[i] = -x[i];
      return std::move(x);
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator+(const std::vector<T>& x, const std::vector<T>& y) {
      const auto n = std::min(std::size(x), std::size(y));
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] + y[i];
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::vector<T>& x, const std::vector<T>& y) {
      const auto n = std::min(std::size(x), std::size(y));
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] - y[i];
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const T& val, const std::vector<T>& x) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = val * x[i];
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const T& val, std::vector<T>&& x) {
      const auto n = std::size(x);
      for (std::size_t i = 0; i < n; ++i) x[i] = val * x[i];
      return std::move(x);
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const std::vector<T>& x, const T& val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] * val;
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(std::vector<T>&& x, const T& val) {
      const auto n = std::size(x);
      for (std::size_t i = 0; i < n; ++i) x[i] = x[i] * val;
      return std::move(x);
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(const std::vector<T>& x, const T& val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] / val;
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(std::vector<T>&& x, const T& val) {
      const auto n = std::size(x);
      for (std::size_t i = 0; i < n; ++i) x[i] = x[i] / val;
      return std::move(x);
    }
  } // namespace arithmetic_ops

  /// @}
} // namespace kspc
