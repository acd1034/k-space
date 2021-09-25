/// @file linalg.hpp
#pragma once
#include <array>
#include <complex>
#include <vector>
#include <kspc/math.hpp>

namespace kspc {
  namespace lapack {
    extern "C" {
    void zheev_(const char& JOBZ, const char& UPLO, const int& N, std::complex<double>* A,
                const int& LDA, double* W, std::complex<double>* WORK, const int& LWORK,
                double* RWORK, int& INFO, int JOBZlen, int UPLOlen);
    }
  } // namespace lapack

  /// diagonalize hermitian matrix
  // clang-format off
  template <typename M,
            std::enable_if_t<std::conjunction_v<
              is_range<M>,
              is_complex<std::decay_t<range_reference_t<M>>>,
              is_fixed_size_array<M>>, std::nullptr_t> = nullptr>
  // clang-format on
  auto zheev(const M& H) {
    using T = complex_value_t<std::decay_t<range_reference_t<M>>>;
    constexpr auto N = isqrt(fixed_size_array_size_v<M>);
    static std::complex<T> tmp_H[N * N];
    static T tmp_E[N];
    static std::complex<T> cwork[N * 4];
    static T rwork[N * 4];
    int info;

    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < N; ++k) {
        tmp_H[k * N + j] = H[j * N + k];
      }
    }

    lapack::zheev_('V', 'L', N, tmp_H, N, tmp_E, cwork, N * 4, rwork, info, 1, 1);

    std::array<T, N> E{};
    std::array<std::array<std::complex<T>, N>, N> U{};
    for (std::size_t j = 0; j < N; ++j) {
      E[j] = tmp_E[j];
      for (std::size_t k = 0; k < N; ++k) {
        // U[j] is an eigenvector of H with eigenvalue E[j]
        U[j][k] = tmp_H[j * N + k];
      }
    }

    return std::make_pair(std::move(E), std::move(U));
  }

  /// @overload
  // clang-format off
  template <typename M,
            std::enable_if_t<std::conjunction_v<
              is_range<M>,
              is_complex<std::decay_t<range_reference_t<M>>>,
              std::negation<is_fixed_size_array<M>>>, std::nullptr_t> = nullptr>
  // clang-format on
  auto zheev(const M& H) {
    using T = complex_value_t<std::decay_t<range_reference_t<M>>>;
    const std::size_t N = isqrt(std::size(H));
    auto* tmp_H = new std::complex<T>[N * N];
    auto* tmp_E = new T[N];
    auto* cwork = new std::complex<T>[N * 4];
    auto* rwork = new T[N * 4];
    int info;

    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < N; ++k) {
        tmp_H[k * N + j] = H[j * N + k];
      }
    }

    lapack::zheev_('V', 'L', N, tmp_H, N, tmp_E, cwork, N * 4, rwork, info, 1, 1);

    std::vector<T> E(N);
    std::vector<std::vector<std::complex<T>>> U(N, std::vector<std::complex<T>>(N));
    for (std::size_t j = 0; j < N; ++j) {
      E[j] = tmp_E[j];
      for (std::size_t k = 0; k < N; ++k) {
        // U[j] is an eigenvector of H with eigenvalue E[j]
        U[j][k] = tmp_H[j * N + k];
      }
    }

    delete[] tmp_H;
    delete[] tmp_E;
    delete[] cwork;
    delete[] rwork;
    return std::make_pair(std::move(E), std::move(U));
  }
} // namespace kspc
