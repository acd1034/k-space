/// @file kspc.hpp
#pragma once
#include <array>
#include <complex>
#include <vector>
#include <gsl/gsl_integration.h>
#include <kspc/core.hpp>

namespace kspc {
  namespace lapack {
    extern "C" {
    void zheev_(const char& JOBZ, const char& UPLO, const int& N, std::complex<double>* A,
                const int& LDA, double* W, std::complex<double>* WORK, const int& LWORK,
                double* RWORK, int& INFO, int JOBZlen, int UPLOlen);
    }
  } // namespace lapack

  // TODO: 固定長・可変長対応
  template <std::size_t N, typename T>
  auto zheev(const ndmatrix<std::complex<T>>& H) {
    static std::complex<T> tmp_H[N * N];
    static T tmp_E[N];
    static std::complex<T> cwork[N * 4];
    static T rwork[N * 4];
    int info;

    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        tmp_H[j * N + i] = H(i, j);
      }
    }

    lapack::zheev_('V', 'L', N, tmp_H, N, tmp_E, cwork, N * 4, rwork, info, 1, 1);

    std::vector<T> E(N);
    std::vector<std::vector<std::complex<T>>> U(N, std::vector<std::complex<T>>(N));
    for (std::size_t i = 0; i < N; ++i) {
      E[i] = tmp_E[i];
      for (std::size_t j = 0; j < N; ++j) {
        U[i][j] = tmp_H[i * N + j]; // U[i]はHの固有値E[i]の固有ベクトル
      }
    }

    return std::pair{std::move(E), std::move(U)};
  }
} // namespace kspc
