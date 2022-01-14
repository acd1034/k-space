/// @file math_basics.hpp
#pragma once
#include <array>
#include <cmath> // sqrt, round
#include <complex>
#include <vector>
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

  /// fixed_size_matrix_dim_v
  // TODO: add fixed_size_matrix_dim
  template <typename T>
  inline constexpr auto fixed_size_matrix_dim_v = isqrt(fixed_size_array_size<T>::value);

  /// dim
  template <typename M, std::enable_if_t<is_sized_range_v<M>, std::nullptr_t> = nullptr>
  inline auto dim(const M& m) {
    return static_cast<decltype(adl_size(m))>(std::round(std::sqrt(adl_size(m))));
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
      return lda_ * i + j;
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
  inline constexpr auto mapping_column_major(const std::size_t lda) {
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

  /// %complex_traits
  template <typename T>
  struct complex_traits {
    using value_type = T;
  };

  /// partial specialization of `complex_traits`
  template <typename T>
  struct complex_traits<std::complex<T>> {
    using value_type = T;
  };

  /// partial specialization of `complex_traits`
  template <typename T>
  struct complex_traits<const T> : complex_traits<T> {};

  /// complex_value_t
  template <typename T>
  using complex_value_t = typename complex_traits<remove_cvref_t<T>>::value_type;

  /// %conj_fn
  struct conj_fn {
    using is_transparent = void;

    template <typename T, std::enable_if_t<is_complex_v<std::decay_t<T>>, std::nullptr_t> = nullptr>
    constexpr auto operator()(T&& x) const noexcept(noexcept(std::conj(std::forward<T>(x))))
      -> decltype(std::conj(std::forward<T>(x))) {
      return std::conj(std::forward<T>(x));
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

// arithmetic operators for `std::array`, `std::vector`
namespace kspc {
  /// @addtogroup matrix
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

  /// @}
} // namespace kspc
