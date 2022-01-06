/// @file math_basics.hpp
#pragma once
#include <array>
#include <cmath> // sqrt, round
#include <complex>
#include <functional> // invoke
#include <kspc/core.hpp>

// dim
namespace kspc {
  /// @addtogroup matrix
  /// @{

  // fixed-size array support

  /// %fixed_size_array_size
  template <typename>
  struct fixed_size_array_size {};

  /// partial specialization of `fixed_size_array_size`
  template <typename T, std::size_t N>
  struct fixed_size_array_size<T[N]> : std::integral_constant<std::size_t, N> {};

  /// partial specialization of `fixed_size_array_size`
  template <typename T, std::size_t N>
  struct fixed_size_array_size<std::array<T, N>> : std::integral_constant<std::size_t, N> {};

  /// helper variable template for `fixed_size_array_size`
  template <typename T>
  inline constexpr auto fixed_size_array_size_v = fixed_size_array_size<T>::value;

  /// %is_fixed_size_array
  template <typename T, typename = void>
  struct is_fixed_size_array : std::false_type {};

  /// partial specialization of `is_fixed_size_array`
  template <typename T>
  struct is_fixed_size_array<T, std::void_t<decltype(fixed_size_array_size<T>::value)>>
    : std::true_type {};

  /// @brief helper variable template for `is_fixed_size_array`
  /// @details To make this true, partially/fully specialize `fixed_size_array_size`.
  template <typename T>
  inline constexpr bool is_fixed_size_array_v = is_fixed_size_array<T>::value;

  // matrix dimension

  /// compile-time sqrt for unsigned integer
  inline constexpr std::size_t isqrt(const std::size_t n) noexcept {
    std::size_t l = 0, r = n;
    while (r - l > 1) {
      const std::size_t mid = l + (r - l) / 2;
      if (mid * mid <= n) {
        l = mid;
      } else {
        r = mid;
      }
    }
    return l;
  }

  /// dim
  // clang-format off
  template <typename M,
            std::enable_if_t<is_fixed_size_array_v<M>, std::nullptr_t> = nullptr>
  // clang-format on
  inline constexpr auto dim(const M&) noexcept {
    return isqrt(fixed_size_array_size_v<M>);
  }

  /// @overload
  // clang-format off
  template <typename M,
            std::enable_if_t<
              !is_fixed_size_array_v<M> &&
              is_sized_range_v<M>, std::nullptr_t> = nullptr>
  // clang-format on
  inline auto dim(const M& m) noexcept(noexcept(std::round(std::sqrt(adl_size(m)))))
    -> decltype(adl_size(m)) {
    return std::round(std::sqrt(adl_size(m)));
  }

  /// @}
} // namespace kspc

// mapping
namespace kspc {
  /// @addtogroup matrix
  /// @{

  /// %mapping_row_major
  struct mapping_row_major {
  private:
    std::size_t lda_{};

  public:
    using size_type = std::size_t;
    constexpr mapping_row_major() = default;
    constexpr explicit mapping_row_major(const size_type lda) : lda_(lda) {}

    constexpr size_type operator()(const size_type i, const size_type j) const noexcept {
      return lda * i + j;
    }
  }; // struct mapping_row_major

  /// %mapping_transpose
  template <typename Mapping>
  struct mapping_transpose {
  private:
    Mapping mapping_{};

  public:
    using size_type = std::size_t;
    constexpr mapping_transpose() = default;
    constexpr explicit mapping_transpose(const Mapping& mapping) : mapping_(mapping) {}
    constexpr explicit mapping_transpose(Mapping&& mapping) : mapping_(std::move(mapping)) {}

    constexpr size_type operator()(const size_type i, const size_type j) const noexcept {
      return mapping_(j, i);
    }
  }; // struct mapping_transpose

  /// deduction guide for @link mapping_transpose mapping_transpose @endlink
  template <typename Mapping>
  mapping_transpose(Mapping) -> mapping_transpose<Mapping>;

  /// mapping_column_major
  inline constexpr auto mapping_column_major(const size_type lda) {
    return mapping_transpose(mapping_row_major(lda));
  }

  /// @}
} // namespace kspc

// projection
namespace kspc {
  /// @addtogroup matrix
  /// @{

  // identity function

  /// %identity_fn
  struct identity_fn {
    using is_transparent = void;

    template <typename T>
    constexpr T&& operator()(T&& t) const noexcept {
      return std::forward<T>(t);
    }
  }; // struct identity_fn

  inline namespace cpo {
    /// identity
    inline constexpr identity_fn identity{};
  } // namespace cpo

  // consistent complex access

  /// %is_complex
  template <typename T>
  struct is_complex : std::false_type {};

  /// partial specialization of `is_complex`
  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

  /// helper variable template for `is_complex`
  template <typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;

  /// %conj_fn
  struct conj_fn {
    using is_transparent = void;

    template <typename C, std::enable_if_t<is_complex_v<std::decay_t<C>>, std::nullptr_t> = nullptr>
    constexpr auto operator()(C&& x) const noexcept(noexcept(std::conj(std::forward<C>(x))))
      -> decltype(std::conj(std::forward<C>(x))) {
      return std::conj(std::forward<C>(x));
    }

    // same as `identity`
    template <typename T,
              std::enable_if_t<!is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
    constexpr T&& operator()(T&& x) const noexcept {
      return std::forward<T>(x);
    }
  }; // struct conj_fn

  inline namespace cpo {
    /// conj
    inline constexpr conj_fn conj{};
  } // namespace cpo

  /// @}
} // namespace kspc

// Numerical algorithms
namespace kspc {
  /// @addtogroup numeric
  /// @{

  // arithmetic operators for `std::array`, `std::vector`

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
    inline constexpr auto operator*(T val, const std::array<T, N>& x) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = val * x[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const std::array<T, N>& x, T val) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] * val;
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(const std::array<T, N>& x, T val) {
      std::array<T, N> ret{};
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] / val;
      return ret;
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
    inline constexpr auto operator*(T val, const std::vector<T>& x) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = val * x[i];
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const std::vector<T>& x, T val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] * val;
      return ret;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(const std::vector<T>& x, T val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t i = 0; i < n; ++i) ret[i] = x[i] / val;
      return ret;
    }
  } // namespace arithmetic_ops

  // sum

  // clang-format off
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
      using U = std::decay_t<std::invoke_result_t<BOp&, T, projected_t<P, I>>>;
      U ret = std::move(init);
      for(; first != last; ++first)
        ret = std::invoke(bop, std::move(ret), std::invoke(proj, *first));
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
                          BOp1 bop1 = {}, BOp2 bop2 = {}, P1 proj1  = {}, P2 proj2  = {}) {
      using U = std::decay_t<
        std::invoke_result_t<
          BOp1&,
          T,
          std::invoke_result_t<BOp2&, projected_t<P1, I1>, projected_t<P2, I2>>>>;
      U ret = std::move(init);
      for (; first1 != last1 && first2 != last2; ++first1, ++first2)
        ret =
          std::invoke(
            bop1,
            std::move(ret),
            std::invoke(bop2, std::invoke(proj1, *first1), std::invoke(proj2, *first2)));
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
                        P1 proj1  = {}, P2 proj2  = {}, BOp1 bop1 = {}, BOp2 bop2 = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r1) && !empty(r2));

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

  // innerp2: ([1] BOp2 ([2] BOp3 [3])) BOp1 ...

  /// @cond
  namespace detail {
    // NOTE: irrelevant to `std::projected`
    template <typename BOp, typename P1, typename I1, typename P2, typename I2>
    using projected2_t = std::invoke_result_t<BOp&, projected_t<P1, I1>, projected_t<P2, I2>>;

    template <typename I1, typename I2, typename I3, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity_fn,
              typename P3 = identity_fn>
    struct innerp2_constraints : std::conjunction<
      std::is_invocable<P1&, iter_value_t<I1>>,
      std::is_invocable<P2&, iter_value_t<I2>>,
      std::is_invocable<P3&, iter_value_t<I3>>,
      std::is_invocable<BOp3&, projected_t<P2, I2>, projected_t<P3, I3>>,
      std::is_invocable<
        BOp2&,
        projected_t<P1, I1>,
        projected2_t<BOp3, P2, I2, P3, I3>>,
      std::is_invocable<
        BOp1&,
        T,
        std::invoke_result_t<
          BOp2&,
          projected_t<P1, I1>,
          projected2_t<BOp3, P2, I2, P3, I3>>>> {};

    template <typename I1, typename S1, typename I2, typename S2, typename I3, typename S3, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity_fn,
              typename P3 = identity_fn,
              std::enable_if_t<std::conjunction_v<
                is_sentinel_for<S1, I1>,
                is_sentinel_for<S2, I2>,
                is_sentinel_for<S3, I3>,
                is_input_iterator<I1>,
                is_input_iterator<I2>,
                is_input_iterator<I3>,
                innerp2_constraints<I1, I2, I3, T, BOp1, BOp2, BOp3, P1, P2, P3>>, std::nullptr_t> = nullptr>
    constexpr auto
    innerp2(I1 first1, S1 last1, I2 first2, S2 last2, I3 first3, S3 last3, T init,
            BOp1 bop1 = {}, BOp2 bop2 = {}, BOp3 bop3 = {}, P1 proj1 = {}, P2 proj2 = {}, P3 proj3 = {}) {
      using U = std::decay_t<
        std::invoke_result_t<
          BOp1&,
          T,
          std::invoke_result_t<
            BOp2&,
            projected_t<P1, I1>,
            projected2_t<BOp3, P2, I2, P3, I3>>>>;
      U ret = std::move(init);
      for (; first1 != last1; ++first1) {
        for (auto first3_copy = first3;
             first2 != last2 && first3_copy != last3;
             ++first2, ++first3_copy)
          ret =
            std::invoke(
              bop1,
              std::move(ret),
              std::invoke(
                bop2,
                std::invoke(proj1, *first1),
                std::invoke(
                  bop3,
                  std::invoke(proj2, *first2),
                  std::invoke(proj3, *first3_copy))));
      }
      return ret;
    }
  } // namespace detail
  /// @endcond

  /// `innerp` with matrix
  // WORKAROUND: difficulty in omitting the initial value
  // TODO: SFINAE is incomplete.
  template <typename R1, typename R2, typename R3,
            typename P1 = conj_fn,
            typename P2 = identity_fn,
            typename P3 = identity_fn,
            typename BOp1 = std::plus<>,
            typename BOp2 = std::multiplies<>,
            typename BOp3 = std::multiplies<>,
            typename T = std::common_type_t<
              detail::projected_t<P1, iterator_t<R1>>,
              detail::projected_t<P2, iterator_t<R2>>,
              detail::projected_t<P3, iterator_t<R3>>>,
            std::enable_if_t<std::conjunction_v<
              is_range<R1>,
              is_range<R2>,
              is_range<R3>,
              std::is_default_constructible<T>>, std::nullptr_t> = nullptr>
  constexpr auto
  innerp(R1&& r1, R2&& r2, R3&& r3,
         P1 proj1 = {}, P2 proj2 = {}, P3 proj3 = {}, BOp1 bop1 = {}, BOp2 bop2 = {}, BOp3 bop3 = {}) {
    using std::begin, std::end; // for ADL
    return detail::innerp2(
             begin(r1), end(r1), begin(r2), end(r2), begin(r3), end(r3), T{},
             std::move(bop1),
             std::move(bop2),
             std::move(bop3),
             std::move(proj1),
             std::move(proj2),
             std::move(proj3));
  }
  // clang-format on

  /// @}
} // namespace kspc
