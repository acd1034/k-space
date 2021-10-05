/// @file linalg.hpp
#pragma once
#include <array>
#include <complex>
#include <vector>
#include <kspc/math_basics.hpp>

namespace kspc {
  /// @addtogroup linalg
  /// @{

  /// @cond
  namespace lapack {
    extern "C" {
    void zheev_(const char& JOBZ, const char& UPLO, const int& N, std::complex<double>* A,
                const int& LDA, double* W, std::complex<double>* WORK, const int& LWORK,
                double* RWORK, int& INFO);
    }
  } // namespace lapack
    /// @endcond

  /// @brief diagonalize hermitian matrix (fixed-size)
  /// @example haldane_fixed.cpp
  // clang-format off
  template <typename L,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<L>,
              is_complex<std::decay_t<range_reference_t<L>>>,
              is_fixed_size_array<L>>, std::nullptr_t> = nullptr>
  // clang-format on
  auto zheev(const L& H) {
    using T = complex_value_t<std::decay_t<range_reference_t<L>>>;
    constexpr auto N = isqrt(fixed_size_array_size_v<L>);
    static std::complex<T> temp_H[N * N];
    static T temp_E[N];
    static std::complex<T> cwork[N * 4];
    static T rwork[N * 4];
    int info;

    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < N; ++k) {
        temp_H[k * N + j] = H[j * N + k];
      }
    }

    lapack::zheev_('V', 'L', N, temp_H, N, temp_E, cwork, N * 4, rwork, info);

    std::array<T, N> E{};
    std::array<std::array<std::complex<T>, N>, N> U{};
    for (std::size_t j = 0; j < N; ++j) {
      E[j] = temp_E[j];
      for (std::size_t k = 0; k < N; ++k) {
        // U[j] is an eigenvector of H with eigenvalue E[j]
        U[j][k] = temp_H[j * N + k];
      }
    }

    return std::make_pair(std::move(E), std::move(U));
  }

  /// @brief diagonalize hermitian matrix (dynamic-size)
  /// @example haldane.cpp
  // clang-format off
  template <typename M,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<M>,
              is_complex<std::decay_t<range_reference_t<M>>>,
              std::negation<is_fixed_size_array<M>>>, std::nullptr_t> = nullptr>
  // clang-format on
  auto zheev(const M& H) {
    using T = complex_value_t<std::decay_t<range_reference_t<M>>>;
    const std::size_t N = kspc::dim(H);
    auto* temp_H = new std::complex<T>[N * N];
    auto* temp_E = new T[N];
    auto* cwork = new std::complex<T>[N * 4];
    auto* rwork = new T[N * 4];
    int info;

    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < N; ++k) {
        temp_H[k * N + j] = H[j * N + k];
      }
    }

    lapack::zheev_('V', 'L', N, temp_H, N, temp_E, cwork, N * 4, rwork, info);

    std::vector<T> E(N);
    std::vector<std::vector<std::complex<T>>> U(N, std::vector<std::complex<T>>(N));
    for (std::size_t j = 0; j < N; ++j) {
      E[j] = temp_E[j];
      for (std::size_t k = 0; k < N; ++k) {
        // U[j] is an eigenvector of H with eigenvalue E[j]
        U[j][k] = temp_H[j * N + k];
      }
    }

    delete[] temp_H;
    delete[] temp_E;
    delete[] cwork;
    delete[] rwork;
    return std::make_pair(std::move(E), std::move(U));
  }

  // @}
} // namespace kspc
