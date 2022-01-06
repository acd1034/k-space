/// @file math.hpp
#pragma once
#include <algorithm> // all_of, max
#include <array>
#include <cmath> // abs, isinf, exp, pow, cosh, sinh, etc.
#include <complex>
#include <functional> // invoke
#include <vector>
#include <kspc/math_basics.hpp>

namespace kspc {
  // matrix element

  /// @cond
  namespace detail {
    // clang-format off
    template <typename M, typename Vs>
    struct fixed_mel_constraints : std::conjunction<
      is_fixed_size_array<M>,
      is_fixed_size_array<Vs>,
      is_fixed_size_array<range_reference_t<Vs>>> {};
    // clang-format on
  } // namespace detail
    /// @endcond

  /// mel
  // clang-format off
  template <typename L, typename Vs,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<L>,
              is_range<Vs>,
              is_range<range_reference_t<Vs>>,
              detail::fixed_mel_constraints<L, Vs>>, std::nullptr_t> = nullptr>
  // clang-format on
  constexpr auto mel(const L& op, const Vs& vs) {
    constexpr std::size_t N = fixed_size_array_size_v<Vs>;
    static_assert(fixed_size_array_size_v<L> == N * N
                    && fixed_size_array_size_v<remove_cvref_t<range_reference_t<Vs>>> == N,
                  "The matrix size and the vector sizes are inconsistent.");

    using T = std::common_type_t<range_reference_t<L>, range_reference_t<range_reference_t<Vs>>>;
    matrix<T, N> ret{};
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        ret(i, j) = innerp(vs[i], op, vs[j]);
      }
    }
    return ret;
  }

  /// @overload
  // clang-format off
  template <typename M, typename Vs,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<M>,
              is_range<Vs>,
              is_range<range_reference_t<Vs>>,
              std::negation<detail::fixed_mel_constraints<M, Vs>>>, std::nullptr_t> = nullptr>
  // clang-format on
  constexpr auto mel(const M& op, const Vs& vs) {
    using std::size; // for ADL
    const std::size_t N = kspc::dim(op);
    assert(size(vs) == N);

    using T = std::common_type_t<range_reference_t<M>, range_reference_t<range_reference_t<Vs>>>;
    ndmatrix<T> ret(N);
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        ret(i, j) = innerp(vs[i], op, vs[j]);
      }
    }
    return ret;
  }

  /// @}

  /// @addtogroup math_constants
  /// @{

  // mathematical constants

  /// @cond
  namespace detail {
    template <typename T>
    using enable_if_floating = std::enable_if_t<std::is_floating_point_v<T>, T>;
  } // namespace detail
    /// @endcond

  // clang-format off
  /// Imaginary unit
  template <typename T> inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, std::complex<T>> i_v{0, 1};
  /// The Euler's number
  template <typename T> inline constexpr T e_v          = detail::enable_if_floating<T>(2.718281828459045235360287471352662498L);
  /// log_2 e
  template <typename T> inline constexpr T log2e_v      = detail::enable_if_floating<T>(1.442695040888963407359924681001892137L);
  /// log_10 e
  template <typename T> inline constexpr T log10e_v     = detail::enable_if_floating<T>(0.434294481903251827651128918916605082L);
  /// log_e 2
  template <typename T> inline constexpr T ln2_v        = detail::enable_if_floating<T>(0.693147180559945309417232121458176568L);
  /// log_e 10
  template <typename T> inline constexpr T ln10_v       = detail::enable_if_floating<T>(2.302585092994045684017991454684364208L);
  /// pi
  template <typename T> inline constexpr T pi_v         = detail::enable_if_floating<T>(3.141592653589793238462643383279502884L);
  /// tau
  template <typename T> inline constexpr T tau_v        = detail::enable_if_floating<T>(6.283185307179586476925286766559005768L);
  /// sqrt 2
  template <typename T> inline constexpr T sqrt2_v      = detail::enable_if_floating<T>(1.414213562373095048801688724209698079L);
  /// sqrt 3
  template <typename T> inline constexpr T sqrt3_v      = detail::enable_if_floating<T>(1.732050807568877293527446341505872367L);
  /// sqrt 5
  template <typename T> inline constexpr T sqrt5_v      = detail::enable_if_floating<T>(2.236067977499789696409173668731276235L);
  /// The Euler-Mascheroni constant
  template <typename T> inline constexpr T egamma_v     = detail::enable_if_floating<T>(0.577215664901532860606512090082402431L);
  /// Exponential of the Euler-Mascheroni constant
  template <typename T> inline constexpr T exp_egamma_v = detail::enable_if_floating<T>(1.781072417990197985236504103107179549L);

  /// alias for `i_v<double>`
  inline constexpr std::complex<double> i = i_v<double>;
  /// alias for `e_v<double>`
  inline constexpr double e          = e_v<double>;
  /// alias for `log2e_v<double>`
  inline constexpr double log2e      = log2e_v<double>;
  /// alias for `log10e_v<double>`
  inline constexpr double log10e     = log10e_v<double>;
  /// alias for `ln2_v<double>`
  inline constexpr double ln2        = ln2_v<double>;
  /// alias for `ln10_v<double>`
  inline constexpr double ln10       = ln10_v<double>;
  /// alias for `pi_v<double>`
  inline constexpr double pi         = pi_v<double>;
  /// alias for `tau_v<double>`
  inline constexpr double tau        = tau_v<double>;
  /// alias for `sqrt2_v<double>`
  inline constexpr double sqrt2      = sqrt2_v<double>;
  /// alias for `sqrt3_v<double>`
  inline constexpr double sqrt3      = sqrt3_v<double>;
  /// alias for `sqrt5_v<double>`
  inline constexpr double sqrt5      = sqrt5_v<double>;
  /// alias for `egamma_v<double>`
  inline constexpr double egamma     = egamma_v<double>;
  /// alias for `exp_egamma_v<double>`
  inline constexpr double exp_egamma = exp_egamma_v<double>;
  // clang-format on

  /// @}

  /// @addtogroup math_functions
  /// @{

  // mathematical functions

  /// The Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> f_FD(const T ene, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (ene - mu)) + 1.0);
  }

  /// Derivative of the Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> Df_FD(const T ene, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::cosh(0.5 * beta * (ene - mu)), -2);
  }

  /// The Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> f_BE(const T ene, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (ene - mu)) - 1.0);
  }

  /// Derivative of the Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> Df_BE(const T ene, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::sinh(0.5 * beta * (ene - mu)), -2);
  }

  /// lerp (C++20)
  template <typename T, typename U>
  inline constexpr auto lerp(const T& a, const T& b, const U& t) noexcept(noexcept(a + t * (b - a)))
    -> decltype((a + t * (b - a))) {
    if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<U>) {
      if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) {
        return (1 - t) * a + t * b;
      }
    }
    return a + t * (b - a);
  }

  /// squared
  template <typename T>
  inline constexpr auto squared(const T& x) noexcept(noexcept(x* x)) -> decltype((x * x)) {
    return x * x;
  }

  /// cubed
  template <typename T>
  inline constexpr auto cubed(const T& x) noexcept(noexcept(x* x* x)) -> decltype((x * x * x)) {
    return x * x * x;
  }

  /// @}

  /// @addtogroup physics
  /// @{

  // in Brillouin zone

  template <typename K, typename B,
            std::enable_if_t<is_range_v<K> && is_range_v<B>, std::nullptr_t> = nullptr>
  inline constexpr bool in_BZ(const K& k, const B& b) {
    return 2 * std::abs(innerp(k, b)) < innerp(b, b);
  }

  template <typename Bs, std::enable_if_t<is_range_v<Bs>, std::nullptr_t> = nullptr>
  inline constexpr auto make_in_BZ(const Bs& bs) {
    return [&bs](const auto& k) {
      using std::begin, std::end; // for ADL
      return std::all_of(begin(bs), end(bs), [&k](const auto& b) { return in_BZ(k, b); });
    };
  }

  /// @}
} // namespace kspc
