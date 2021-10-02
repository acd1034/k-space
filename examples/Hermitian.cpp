/**
 * @file Hermitian.cpp
 * compiler: GCC version 10.2.0
 * compiler option:
 * -O3 -std=c++17 -lm -llapack -lblas -lgsl -lgslcblas -mtune=native -march=native -mfpmath=both
 */
#include <fstream>
#include <iomanip> // boolalpha
#include <iostream>
#include <kspc/math.hpp>

int main() {
  const auto identity = kspc::identity{};
  const auto conj = kspc::conj_fn{};
  const auto negate = std::negate<>{};
  std::cout << std::boolalpha << std::endl;
  // clang-format off
  {
    // Check whether matrix is hermitian.
    kspc::matrix<std::complex<double>, 2> m{
      2.0,            1.0 - kspc::i,
      1.0 + kspc::i, -2.0,
    };
    std::cout << kspc::hermitian(m) << std::endl;
  }
  {
    // `hermitian` can also be used for checking whether real matrix is symmetric.
    kspc::matrix<double, 2> m{
      2.0,  1.0,
      1.0, -2.0,
    };
    std::cout << kspc::hermitian(m) << std::endl;
  }
  {
    // To check whether complex matrix is symmetric, pass `identity` for the first argument.
    kspc::matrix<std::complex<double>, 2> m{
      2.0,            1.0 + kspc::i,
      1.0 + kspc::i, -2.0,
    };
    std::cout << kspc::hermitian(m, identity) << std::endl;
  }
  {
    // Check whether matrix is skew-hermitian.
    // conj(m(j, k)) == -m(k, j) for all j, k.
    kspc::matrix<std::complex<double>, 2> m{
      2.0 * kspc::i, -1.0 + kspc::i,
      1.0 + kspc::i, -2.0 * kspc::i,
    };
    std::cout << kspc::hermitian(m, conj, negate) << std::endl;
  }
  {
    // Check whether matrix is antisymmetric.
    // m(j, k) == -m(k, j) for all j, k.
    kspc::matrix<double, 2> m{
      0.0, -1.0,
      1.0,  0.0,
    };
    std::cout << kspc::hermitian(m, identity, negate) << std::endl;
  }
  // clang-format on
}
