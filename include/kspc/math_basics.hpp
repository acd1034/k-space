/// @file math_basics.hpp
#pragma once
#include <array>
#include <vector>
#include <kspc/core.hpp>

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
