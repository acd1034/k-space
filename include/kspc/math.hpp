/// @file math.hpp
#pragma once
#include <algorithm> // all_of
#include <cmath>     // abs, exp, pow, cosh, sinh, etc.
#include <complex>
#include <kspc/core.hpp>    // is_range
#include <kspc/numeric.hpp> // innerp

// Mathematical constants
namespace kspc {
  /// @addtogroup math_constants
  /// @{

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
  template <typename T> inline constexpr detail::enable_if_floating<T>
  e_v          = T(2.718281828459045235360287471352662498L);
  /// log_2 e
  template <typename T> inline constexpr detail::enable_if_floating<T>
  log2e_v      = T(1.442695040888963407359924681001892137L);
  /// log_10 e
  template <typename T> inline constexpr detail::enable_if_floating<T>
  log10e_v     = T(0.434294481903251827651128918916605082L);
  /// log_e 2
  template <typename T> inline constexpr detail::enable_if_floating<T>
  ln2_v        = T(0.693147180559945309417232121458176568L);
  /// log_e 10
  template <typename T> inline constexpr detail::enable_if_floating<T>
  ln10_v       = T(2.302585092994045684017991454684364208L);
  /// pi
  template <typename T> inline constexpr detail::enable_if_floating<T>
  pi_v         = T(3.141592653589793238462643383279502884L);
  /// tau
  template <typename T> inline constexpr detail::enable_if_floating<T>
  tau_v        = T(6.283185307179586476925286766559005768L);
  /// sqrt 2
  template <typename T> inline constexpr detail::enable_if_floating<T>
  sqrt2_v      = T(1.414213562373095048801688724209698079L);
  /// sqrt 3
  template <typename T> inline constexpr detail::enable_if_floating<T>
  sqrt3_v      = T(1.732050807568877293527446341505872367L);
  /// sqrt 5
  template <typename T> inline constexpr detail::enable_if_floating<T>
  sqrt5_v      = T(2.236067977499789696409173668731276235L);
  /// The Euler-Mascheroni constant
  template <typename T> inline constexpr detail::enable_if_floating<T>
  egamma_v     = T(0.577215664901532860606512090082402431L);
  /// Exponential of the Euler-Mascheroni constant
  template <typename T> inline constexpr detail::enable_if_floating<T>
  exp_egamma_v = T(1.781072417990197985236504103107179549L);

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
} // namespace kspc

// Mathematical functions
namespace kspc {
  /// @addtogroup math_functions
  /// @{

  /// The Fermi distribution
  template <typename T>
  detail::enable_if_floating<T> ffermi(const T ene, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (ene - mu)) + 1.0);
  }

  /// Derivative of the Fermi distribution
  template <typename T>
  detail::enable_if_floating<T> dffermi(const T ene, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::cosh(0.5 * beta * (ene - mu)), -2);
  }

  /// The Bose distribution
  template <typename T>
  detail::enable_if_floating<T> fbose(const T ene, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (ene - mu)) - 1.0);
  }

  /// Derivative of the Bose distribution
  template <typename T>
  detail::enable_if_floating<T> dfbose(const T ene, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::sinh(0.5 * beta * (ene - mu)), -2);
  }

  /// squared
  template <typename T>
  inline constexpr auto squared(const T& x) noexcept(noexcept(x* x)) -> decltype(x * x) {
    return x * x;
  }

  /// cubed
  template <typename T>
  inline constexpr auto cubed(const T& x) noexcept(noexcept(x* x* x)) -> decltype(x * x * x) {
    return x * x * x;
  }

  /// @}
} // namespace kspc

// k-points
namespace kspc::kpts {
  /// @addtogroup physics
  /// @{

  // Primitive hexagonal (HEX)
  namespace hex {
    inline constexpr std::array Gamma{0.0, 0.0, 0.0};
    inline constexpr std::array M{0.5, 0.5, 0.0};
    inline constexpr std::array K{2.0 / 3.0, 1.0 / 3.0, 0.0};
    inline constexpr std::array A{0.0, 0.0, 0.5};
    inline constexpr std::array L{0.5, 0.5, 0.5};
    inline constexpr std::array H{2.0 / 3.0, 1.0 / 3.0, 0.5};
  } // namespace hex

  /// @}
} // namespace kspc::kpts

// in_brillouin_zone
namespace kspc {
  /// @addtogroup physics
  /// @{

  /// %in_brillouin_zone
  template <typename Bs>
  struct in_brillouin_zone {
  private:
    Bs bs_{};

  public:
    constexpr in_brillouin_zone() = default;
    constexpr explicit in_brillouin_zone(const Bs& bs) : bs_{bs} {}
    constexpr explicit in_brillouin_zone(Bs&& bs) : bs_{std::move(bs)} {}

    template <typename K>
    constexpr std::enable_if_t<std::conjunction_v<is_range<K>, is_range<Bs>>, bool>
    operator()(const K& k) const {
      using std::begin, std::end; // for ADL
      return std::all_of(begin(bs_), end(bs_),
                         [&k](const auto& b) { return 2 * std::abs(innerp(k, b)) < innerp(b, b); });
    }
  }; // struct in_brillouin_zone

  /// deduction guide for @link in_brillouin_zone in_brillouin_zone @endlink
  template <typename Bs>
  in_brillouin_zone(Bs) -> in_brillouin_zone<Bs>;

  /// @}
} // namespace kspc
