/// @file math.hpp
#pragma once
#include <array>
#include <complex>
#include <limits>
#include <kspc/core.hpp>

namespace kspc {
  // conversion between signed integer and unsigned integer

  /// make_signed_v
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>, std::nullptr_t> = nullptr>
  inline constexpr auto
  make_signed_v(const T& x) noexcept(noexcept(static_cast<std::make_signed_t<T>>(x))) {
    using U = std::make_signed_t<T>;
    assert(x <= static_cast<T>(std::numeric_limits<U>::max()));
    return static_cast<U>(x);
  }

  /// @overload
  template <typename T, std::enable_if_t<!std::is_unsigned_v<T>, std::nullptr_t> = nullptr>
  inline constexpr T&& make_signed_v(T&& x) noexcept(noexcept(std::forward<T>(x))) {
    return std::forward<T>(x);
  }

  /// make_unsigned_v
  template <typename T, std::enable_if_t<std::is_signed_v<T>, std::nullptr_t> = nullptr>
  inline constexpr auto
  make_unsigned_v(const T& x) noexcept(noexcept(static_cast<std::make_unsigned_t<T>>(x))) {
    assert(x >= static_cast<T>(0));
    return static_cast<std::make_unsigned_t<T>>(x);
  }

  /// @overload
  template <typename T, std::enable_if_t<!std::is_signed_v<T>, std::nullptr_t> = nullptr>
  inline constexpr T&& make_unsigned_v(T&& x) noexcept(noexcept(std::forward<T>(x))) {
    return std::forward<T>(x);
  }

  // seemless use of floating-point and complex

  /// %is_complex
  template <typename T>
  struct is_complex : std::false_type {};

  /// partial specialization of `is_complex`
  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

  /// helper variable template for `is_complex`
  template <typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;

  /// complex_value_t
  template <typename T, std::enable_if_t<is_complex_v<T>, std::nullptr_t> = nullptr>
  using complex_value_t = typename T::value_type;

  /// real
  template <typename T, std::enable_if_t<is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr auto real(T&& x) noexcept(noexcept(std::real(std::forward<T>(x))))
    -> decltype((std::real(std::forward<T>(x)))) {
    return std::real(std::forward<T>(x));
  }

  /// @overload
  template <typename T, std::enable_if_t<!is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr T&& real(T&& x) noexcept(noexcept(std::forward<T>(x))) {
    return std::forward<T>(x);
  }

  /// imag
  template <typename T, std::enable_if_t<is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr auto imag(T&& x) noexcept(noexcept(std::imag(std::forward<T>(x))))
    -> decltype((std::imag(std::forward<T>(x)))) {
    return std::imag(std::forward<T>(x));
  }

  /// @overload
  // clang-format off
  template <typename T,
            std::enable_if_t<std::conjunction_v<
              std::negation<is_complex<std::decay_t<T>>>,
              std::is_default_constructible<std::decay_t<T>>>, std::nullptr_t> = nullptr>
  // clang-format on
  inline constexpr auto imag(T&& x) noexcept(noexcept(std::decay_t<T>{})) {
    return std::decay_t<T>{};
  }

  /// conj
  template <typename T, std::enable_if_t<is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr auto conj(T&& x) noexcept(noexcept(std::conj(std::forward<T>(x))))
    -> decltype((std::conj(std::forward<T>(x)))) {
    return std::conj(std::forward<T>(x));
  }

  /// @overload
  template <typename T, std::enable_if_t<!is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr T&& conj(T&& x) noexcept(noexcept(std::forward<T>(x))) {
    return std::forward<T>(x);
  }

  /// %conj_fn
  struct conj_fn {
    template <typename T>
    constexpr auto operator()(T&& t) const noexcept(noexcept(kspc::conj(std::forward<T>(t))))
      -> decltype((kspc::conj(std::forward<T>(t)))) {
      return kspc::conj(std::forward<T>(t));
    }
  }; // struct conj_fn

  // fixed-size array optimization

  /// sqrt for unsigned integer
  inline constexpr std::size_t isqrt(const std::size_t N) {
    std::size_t l = 0, r = N;
    while (r - l > 1) {
      const std::size_t mid = l + (r - l) / 2;
      if (mid * mid <= N) l = mid;
      else
        r = mid;
    }
    return l;
  }

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

  /// helper variable template for `is_fixed_size_array`
  /// To make this true, partially/fully specialize `fixed_size_array_size`.
  template <typename T>
  inline constexpr bool is_fixed_size_array_v = is_fixed_size_array<T>::value;

  // sum

  // clang-format off
  namespace detail {
    // NOTE: irrelevant to `std::projected`
    template <typename P, typename I>
    using projected_t = std::invoke_result_t<P&, iter_reference_t<I>>;

    template <typename I, typename T,
              typename BOp = std::plus<>,
              typename P = identity>
    struct sum_constraints : std::conjunction<
      std::is_invocable<P&, iter_reference_t<I>>,
      std::is_invocable<BOp&, T, projected_t<P, I>>> {};

    template <typename I, typename S, typename T,
              typename BOp = std::plus<>,
              typename P = identity,
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

  /// sum
  /// almost same as `std::accumulate` but does not require initial value.
  /// NOTE: the order of the arguments `P`, `BOp` is different from range-v3.
  template <typename R,
            typename P = identity,
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

  namespace detail {
    template <typename I1, typename I2, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity>
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
              typename P2 = identity,
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

  /// innerp
  /// almost same as `std::inner_product` but does not require initial value.
  /// NOTE: the order of the arguments `PN`, `BOpN` is different from range-v3.
  template <typename R1, typename R2,
            typename P1 = conj_fn,
            typename P2 = identity,
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

  namespace detail {
    // NOTE: irrelevant to `std::projected`
    template <typename BOp, typename P1, typename I1, typename P2, typename I2>
    using projected2_t = std::invoke_result_t<BOp&, projected_t<P1, I1>, projected_t<P2, I2>>;

    template <typename I1, typename I2, typename I3, typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity>
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
              typename P2 = identity,
              typename P3 = identity,
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

  /// `innerp` with matrix
  /// WORKAROUND: difficulty in omitting the initial value.
  // TODO: SFINAE incomplete
  template <typename R1, typename R2, typename R3,
            typename P1 = conj_fn,
            typename P2 = identity,
            typename P3 = identity,
            typename BOp1 = std::plus<>,
            typename BOp2 = std::multiplies<>,
            typename BOp3 = std::multiplies<>,
            typename TImpl = std::invoke_result_t<
              BOp2&,
              detail::projected_t<P1, iterator_t<R1>>,
              detail::projected2_t<BOp3, P2, iterator_t<R2>, P3, iterator_t<R3>>>,
            typename T = std::decay_t<std::invoke_result_t<BOp1&, TImpl, TImpl>>,
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
} // namespace kspc
