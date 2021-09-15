/// @file kspc.hpp
#pragma once
#include <algorithm> // all_of
#include <array>
#include <cmath> // exp, pow, cosh, sinh
#include <complex>
#include <numeric>
#include <vector>
#include <kspc/math.hpp>

namespace kspc {
  // arithmetic operator for `std::array`, `std::vector`

  inline namespace op {
    // array
    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto& operator+(const std::array<T, N>& x) {
      return x;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::array<T, N>& x) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = -x[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator+(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] + y[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator-(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] - y[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(T val, const std::array<T, N>& x) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = val * x[i];
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator*(const std::array<T, N>& x, T val) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] * val;
      return ret;
    }

    template <typename T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    inline constexpr auto operator/(const std::array<T, N>& x, T val) {
      std::array<T, N> ret;
      for (std::size_t i = 0; i < N; ++i) ret[i] = x[i] / val;
      return ret;
    }

    // vector
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
  } // namespace op

  // approximate comparison

  template <typename T>
  inline constexpr T eps_v = std::enable_if_t<std::is_arithmetic_v<T>, T>(1e-6);

  inline constexpr double eps = eps_v<double>;

  template <typename T>
  struct approx {
  private:
    const T val_;
    const double margin_;

  public:
    constexpr explicit approx(const T& val, double margin = eps) noexcept
      : val_(val), margin_(margin) {}
    constexpr explicit approx(T&& val, double margin = eps) noexcept
      : val_(std::move(val)), margin_(margin) {}

    template <typename U>
    friend constexpr bool operator==(const U& x, const approx& y) noexcept {
      // Performs equivalent check of std::fabs(x - y.val_) <= margin_
      // But without the subtraction to allow for INFINITY in comparison
      return (x + y.margin_ >= y.val_) && (y.val_ + y.margin_ >= x);
    }
    template <typename U>
    friend constexpr bool operator<(const U& x, const approx& y) noexcept {
      return !(x == y) && (x < y.val_);
    }
    // boilerplate
    template <typename U>
    constexpr friend bool operator!=(const U& x, const approx& y) noexcept(noexcept(x == y)) {
      return !(x == y);
    }
    template <typename U>
    constexpr friend bool operator>(const U& x, const approx& y) noexcept(noexcept(y < x)) {
      return y < x;
    }
    template <typename U>
    constexpr friend bool operator<=(const U& x, const approx& y) noexcept(noexcept(y < x)) {
      return !(y < x);
    }
    template <typename U>
    constexpr friend bool operator>=(const U& x, const approx& y) noexcept(noexcept(x < y)) {
      return !(x < y);
    }
  }; // struct approx

  struct approx_eq {
    const double margin_ = eps;

    constexpr approx_eq() {}
    constexpr explicit approx_eq(const double margin) : margin_(margin) {}

    template <typename T, typename U>
    constexpr bool operator()(const T& t, const U& u) const {
      return t == approx(u, margin_);
    }
  }; // struct approx_eq

  // TODO: matrix (fixed)

  // ndmatrix

  template <typename T>
  struct ndmatrix {
  private:
    std::size_t dim_ = 0;
    std::vector<T> instance_{};

  public:
    ndmatrix() = default;
    explicit ndmatrix(std::size_t n) : dim_(n), instance_(n * n) {}
    ndmatrix(std::size_t n, const T& value) : dim_(n), instance_(n * n, value) {}
    explicit ndmatrix(std::initializer_list<T> l)
      : dim_(std::size_t(std::sqrt(std::size(l)))), instance_(std::size(l)) {
      std::copy(std::begin(l), std::end(l), std::begin(instance_));
    }
    auto begin() & noexcept(noexcept(instance_.begin())) -> decltype((instance_.begin())) {
      return instance_.begin();
    }
    auto begin() const& noexcept(noexcept(instance_.begin())) -> decltype((instance_.begin())) {
      return instance_.begin();
    }
    auto end() & noexcept(noexcept(instance_.end())) -> decltype((instance_.end())) {
      return instance_.end();
    }
    auto end() const& noexcept(noexcept(instance_.end())) -> decltype((instance_.end())) {
      return instance_.end();
    }

    std::size_t dim() const {
      return dim_;
    }
    auto operator()(std::size_t i, std::size_t j) & noexcept(noexcept(instance_[i * dim() + j]))
      -> decltype((instance_[i * dim() + j])) {
      return instance_[i * dim() + j];
    }
    auto operator()(std::size_t i,
                    std::size_t j) const& noexcept(noexcept(instance_[i * dim() + j]))
      -> decltype((instance_[i * dim() + j])) {
      return instance_[i * dim() + j];
    }
    template <typename Comp = approx_eq, typename Proj1 = conj_fn, typename Proj2 = identity>
    bool is_hermite(Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {}) {
      for (std::size_t i = 0; i < dim(); ++i) {
        for (std::size_t j = i; j < dim(); ++j) {
          if (!std::invoke(comp, std::invoke(proj1, (*this)(i, j)),
                           std::invoke(proj2, (*this)(j, i))))
            return false;
        }
      }
      return true;
    }
  }; // struct ndmatrix

  template <typename T>
  auto mel(const ndmatrix<T>& op, const std::vector<std::vector<T>>& v) {
    const std::size_t n = std::min(op.dim(), std::size(v));
    ndmatrix<T> ret(n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        ret(i, j) = innerp(v[i], op, v[j]);
      }
    }
    return ret;
  }

  // math constants

  namespace detail {
    template <typename T>
    using enable_if_floating = std::enable_if_t<std::is_floating_point_v<T>, T>;
  } // namespace detail

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

  // math functions

  /// The Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> f(const T E, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (E - mu)) + 1.0);
  }

  /// Derivative of the Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> dfdE(const T E, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::cosh(0.5 * beta * (E - mu)), -2);
  }

  /// The Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> n(const T E, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (E - mu)) - 1.0);
  }

  /// Derivative of the Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> dndE(const T E, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::sinh(0.5 * beta * (E - mu)), -2);
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

  // k-space numerical calculation

  template <typename K, typename B,
            std::enable_if_t<is_range_v<K> && is_range_v<B>, std::nullptr_t> = nullptr>
  inline constexpr bool in_BZ(const K& k, const B& b) {
    return 2 * std::abs(innerp(k, b)) < innerp(b, b);
  }

  template <typename Bs, std::enable_if_t<is_range_v<Bs>, std::nullptr_t> = nullptr>
  constexpr auto make_in_BZ(const Bs& bs) {
    return [&bs](const auto& k) {
      using std::begin, std::end; // for ADL
      return std::all_of(begin(bs), end(bs), [&k](const auto& b) { return in_BZ(k, b); });
    };
  }
} // namespace kspc
