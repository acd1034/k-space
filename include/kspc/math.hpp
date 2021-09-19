/// @file math.hpp
#pragma once
#include <complex>
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

  /// is_complex
  template <typename T>
  struct is_complex : std::false_type {};

  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

  /// is_complex_v
  template <typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;

  /// conj
  template <typename T, std::enable_if_t<is_complex_v<remove_cvref_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr auto conj(T&& x) noexcept(noexcept(std::conj(std::forward<T>(x))))
    -> decltype((std::conj(std::forward<T>(x)))) {
    return std::conj(std::forward<T>(x));
  }

  /// @overload
  template <typename T,
            std::enable_if_t<!is_complex_v<remove_cvref_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr T&& conj(T&& x) noexcept(noexcept(std::forward<T>(x))) {
    return std::forward<T>(x);
  }

  /// conj_fn
  struct conj_fn {
    template <typename T>
    constexpr auto operator()(T&& t) const noexcept(noexcept(kspc::conj(std::forward<T>(t))))
      -> decltype((kspc::conj(std::forward<T>(t)))) {
      return kspc::conj(std::forward<T>(t));
    }
  }; // struct conj_fn

  // sum

  // clang-format off
  namespace detail {
    template <typename I,
              typename T,
              typename BOp = std::plus<>,
              typename P = identity>
    struct sum_result
      : std::invoke_result<
                  BOp&,
                  T,
                  std::invoke_result_t<P&, iter_reference_t<I>>> {};

    template <typename I,
              typename T,
              typename BOp = std::plus<>,
              typename P = identity>
    using sum_result_t = typename sum_result<I, T, BOp, P>::type;

    template <typename I, typename S,
              typename T,
              typename BOp = std::plus<>,
              typename P = identity,
              typename U = std::decay_t<sum_result_t<I, T, BOp, P>>>
    constexpr
    std::enable_if_t<std::conjunction_v<
      is_sentinel_for<S, I>,
      is_input_iterator<I>,
      std::is_convertible<T, U>,
      std::is_assignable<U&, sum_result_t<I, U, BOp, P>>>, U>
    sum(I first, S last, T init, BOp bop = {}, P proj = {}) {
      U ret = std::move(init);
      for(; first != last; ++first)
        ret = std::invoke(bop, std::move(ret), std::invoke(proj, *first));
      return ret;
    }
  } // namespace detail

  /// almost same as `std::accumulate` but does not require initial value
  /// NOTE: the order of the arguments `P`, `BOp` is different from range-v3
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
    template <typename I1,
              typename I2,
              typename BOp2 = std::multiplies<>,
              typename P1 = conj_fn, typename P2 = identity>
    struct innerp_result_impl
      : std::invoke_result<
          BOp2&,
          std::invoke_result_t<P1&, iter_reference_t<I1>>,
          std::invoke_result_t<P2&, iter_reference_t<I2>>> {};

    template <typename I1,
              typename I2,
              typename BOp2 = std::multiplies<>,
              typename P1 = conj_fn, typename P2 = identity>
    using innerp_result_impl_t = typename innerp_result_impl<I1, I2, BOp2, P1, P2>::type;

    template <typename I1,
              typename I2,
              typename T,
              typename BOp1 = std::plus<>, typename BOp2 = std::multiplies<>,
              typename P1   = conj_fn,     typename P2   = identity>
    struct innerp_result
      : std::invoke_result<
                  BOp1&,
                  T,
                  innerp_result_impl_t<I1, I2, BOp2, P1, P2>> {};

    template <typename I1,
              typename I2,
              typename T,
              typename BOp1 = std::plus<>, typename BOp2 = std::multiplies<>,
              typename P1   = conj_fn,     typename P2   = identity>
    using innerp_result_t = typename innerp_result<I1, I2, T, BOp1, BOp2, P1, P2>::type;

    template <typename I1, typename S1,
              typename I2, typename S2,
              typename T,
              typename BOp1 = std::plus<>, typename BOp2 = std::multiplies<>,
              typename P1   = conj_fn,     typename P2   = identity,
              typename U = std::decay_t<innerp_result_t<I1, I2, T, BOp1, BOp2, P1, P2>>>
    constexpr
    std::enable_if_t<std::conjunction_v<
      is_sentinel_for<S1, I1>,
      is_sentinel_for<S2, I2>,
      is_input_iterator<I1>,
      is_input_iterator<I2>,
      std::is_convertible<T, U>,
      std::is_assignable<U&, innerp_result_t<I1, I2, U, BOp1, BOp2, P1, P2>>>, U>
    innerp(I1 first1, S1 last1,
           I2 first2, S2 last2,
           T init,
           BOp1 bop1 = {}, BOp2 bop2 = {},
           P1 proj1  = {}, P2 proj2  = {}) {
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

  /// almost same as `std::inner_product` but does not require initial value
  /// NOTE: the order of the arguments `PN`, `BOpN` is different from range-v3
  template <typename R1,
            typename R2,
            typename P1   = conj_fn,     typename P2   = identity,
            typename BOp1 = std::plus<>, typename BOp2 = std::multiplies<>,
            std::enable_if_t<std::conjunction_v<
              is_range<R1>,
              is_range<R2>,
              std::negation<is_range<P1>>>, std::nullptr_t> = nullptr>
  constexpr auto innerp(R1&& r1, R2&& r2,
                        P1 proj1  = {}, P2 proj2  = {},
                        BOp1 bop1 = {}, BOp2 bop2 = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r1) && !empty(r2));

    auto first1 = begin(r1);
    auto first2 = begin(r2);
    auto init = std::invoke(
                  bop2,
                  std::invoke(proj1, *first1++),
                  std::invoke(proj2, *first2++));
    return detail::innerp(
             first1, end(r1),
             first2, end(r2),
             init,
             std::move(bop1),  std::move(bop2),
             std::move(proj1), std::move(proj2));
  }

  // innerp2: ([1] BOp2 ([2] BOp3 [3])) BOp1 ...

  namespace detail {
    template <typename I1, typename I2, typename I3,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity>
    struct innerp2_result_impl
      : std::invoke_result<
          BOp2&,
          std::invoke_result_t<P1&, iter_reference_t<I1>>,
          std::invoke_result_t<
            BOp3&,
            std::invoke_result_t<P2&, iter_reference_t<I2>>,
            std::invoke_result_t<P3&, iter_reference_t<I3>>>> {};

    template <typename I1, typename I2, typename I3,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity>
    using innerp2_result_impl_t = typename innerp2_result_impl<I1, I2, I3, BOp2, BOp3, P1, P2, P3>::type;

    template <typename I1, typename I2, typename I3,
              typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity>
    struct innerp2_result
      : std::invoke_result<
                  BOp1&,
                  T,
                  innerp2_result_impl_t<I1, I2, I3, BOp2, BOp3, P1, P2, P3>> {};

    template <typename I1, typename I2, typename I3,
              typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity>
    using innerp2_result_t = typename innerp2_result<I1, I2, I3, T, BOp1, BOp2, BOp3, P1, P2, P3>::type;

    template <typename I1, typename S1, typename I2, typename S2, typename I3, typename S3,
              typename T,
              typename BOp1 = std::plus<>,
              typename BOp2 = std::multiplies<>,
              typename BOp3 = std::multiplies<>,
              typename P1 = conj_fn,
              typename P2 = identity,
              typename P3 = identity,
              typename U = std::decay_t<innerp2_result_t<I1, I2, I3, T, BOp1, BOp2, BOp3, P1, P2, P3>>>
    constexpr
    std::enable_if_t<std::conjunction_v<
      is_sentinel_for<S1, I1>,
      is_sentinel_for<S2, I2>,
      is_sentinel_for<S3, I3>,
      is_input_iterator<I1>,
      is_input_iterator<I2>,
      is_input_iterator<I3>,
      std::is_convertible<T, U>,
      std::is_assignable<U&, innerp2_result_t<I1, I2, I3, U, BOp1, BOp2, BOp3, P1, P2, P3>>>, U>
    innerp2(I1 first1, S1 last1, I2 first2, S2 last2, I3 first3, S3 last3,
            T init,
            BOp1 bop1 = {},
            BOp2 bop2 = {},
            BOp3 bop3 = {},
            P1 proj1 = {},
            P2 proj2 = {},
            P3 proj3 = {}) {
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
  /// WORKAROUND: difficulty in omitting the initial value
  template <typename R1, typename R2, typename R3,
            typename P1 = conj_fn,
            typename P2 = identity,
            typename P3 = identity,
            typename BOp1 = std::plus<>,
            typename BOp2 = std::multiplies<>,
            typename BOp3 = std::multiplies<>,
            typename T = std::decay_t<std::invoke_result_t<
              BOp1&,
              detail::innerp2_result_impl_t<iterator_t<R1>, iterator_t<R2>, iterator_t<R3>, BOp2, BOp3, P1, P2, P3>,
              detail::innerp2_result_impl_t<iterator_t<R1>, iterator_t<R2>, iterator_t<R3>, BOp2, BOp3, P1, P2, P3>>>,
            std::enable_if_t<std::conjunction_v<
              is_range<R1>,
              is_range<R2>,
              is_range<R3>,
              std::is_default_constructible<T>>, std::nullptr_t> = nullptr>
  constexpr auto innerp(R1&& r1, R2&& r2, R3&& r3,
                        P1 proj1 = {},
                        P2 proj2 = {},
                        P3 proj3 = {},
                        BOp1 bop1 = {},
                        BOp2 bop2 = {},
                        BOp3 bop3 = {}) {
    using std::begin, std::end; // for ADL
    return detail::innerp2(
             begin(r1), end(r1), begin(r2), end(r2), begin(r3), end(r3),
             T{},
             std::move(bop1),
             std::move(bop2),
             std::move(bop3),
             std::move(proj1),
             std::move(proj2),
             std::move(proj3));
  }
  // clang-format on
} // namespace kspc
