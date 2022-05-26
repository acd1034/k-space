#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <array>
#include <complex>
#include <vector>
#include <kspc/approx.hpp>
#include <kspc/core.hpp>
#include <kspc/linalg.hpp>
#include <kspc/math.hpp>
#include <kspc/numeric.hpp>

inline constexpr auto equal_to = [](const auto& x, const auto& y) {
  return kspc::approx::equal_to(x, y, 1e-6);
};

inline constexpr auto equal = [](const auto& x, const auto& y) {
  using std::begin, std::end; // for ADL
  return std::equal(begin(x), end(x), begin(y), end(y), equal_to);
};

TEST_CASE("linalg", "[math][linalg]") {
  { // unitary_transform with row-major dynamic matrix
    using namespace std::complex_literals;
    // clang-format off
    std::vector<std::complex<double>> A{
      2.0i, 4.0 + 4.0i,
      -4.0 + 4.0i, -2.0i,
    };
    constexpr double sqrt3 = kspc::sqrt3;
    std::vector<std::complex<double>> U{
      -1.0 / sqrt3, (1.0 - 1.0i) / sqrt3,
      (1.0 + 1.0i) / sqrt3, 1.0 / sqrt3,
    };
    // clang-format on
    const auto row_major = kspc::mapping::row_major(kspc::dim(A));
    kspc::unitary_transform(A, U, row_major, row_major);
    CHECK(equal(A, std::vector<std::complex<double>>{-6.0i, 0.0, 0.0, 6.0i}));
  }
  { // unitary_transform with row-major static matrix
    using namespace std::complex_literals;
    constexpr std::size_t N = 2;
    // clang-format off
    std::array<std::complex<double>, N * N> A{
      2.0i, 4.0 + 4.0i,
      -4.0 + 4.0i, -2.0i,
    };
    constexpr double sqrt3 = kspc::sqrt3;
    std::array<std::complex<double>, N * N> U{
      -1.0 / sqrt3, (1.0 - 1.0i) / sqrt3,
      (1.0 + 1.0i) / sqrt3, 1.0 / sqrt3,
    };
    // clang-format on
    constexpr auto row_major = kspc::mapping::row_major(N);
    kspc::unitary_transform(A, U, row_major, row_major);
    CHECK(equal(A, std::array<std::complex<double>, N * N>{-6.0i, 0.0, 0.0, 6.0i}));
  }
  { // matrix_vector_solve with column-major dynamic matrix
    // clang-format off
    // NOTE: A is column-major
    std::vector A{
       2.0,  2.0,  1.0,
       1.0, -1.0, -1.0,
      -3.0, -1.0, -2.0,
    };
    // clang-format on
    std::vector<std::size_t> ipiv(3);
    std::vector b{-2.0, -2.0, -5.0};
    const auto info = kspc::matrix_vector_solve(A, ipiv, b);
    CHECK(info == 0);
    CHECK(equal(b, std::vector{1.0, 2.0, 2.0}));
  }
  { // matrix_vector_solve with row-major dynamic matrix
    // clang-format off
    std::vector A{
      2.0,  1.0, -3.0,
      2.0, -1.0, -1.0,
      1.0, -1.0, -2.0,
    };
    // clang-format on
    std::vector b{-2.0, -2.0, -5.0};
    const auto row_major = kspc::mapping::row_major(kspc::dim(A));
    const auto info = kspc::matrix_vector_solve(A, b, row_major);
    CHECK(info == 0);
    CHECK(equal(b, std::vector{1.0, 2.0, 2.0}));
  }
  { // matrix_vector_solve with row-major static matrix
    // clang-format off
    std::array A{
      2.0,  1.0, -3.0,
      2.0, -1.0, -1.0,
      1.0, -1.0, -2.0,
    };
    // clang-format on
    std::array b{-2.0, -2.0, -5.0};
    constexpr auto N = kspc::fixed_size_matrix_dim_v<std::remove_cv_t<decltype(A)>>;
    constexpr auto row_major = kspc::mapping::row_major(N);
    const auto info = kspc::matrix_vector_solve(A, b, row_major);
    CHECK(info == 0);
    CHECK(equal(b, std::array{1.0, 2.0, 2.0}));
  }
  { // hermitian::eigen_solve with column-major dynamic matrix
    using namespace std::complex_literals;
    // clang-format off
    // NOTE: A is column-major
    std::vector<std::complex<double>> A{
      2.0, 1.0 - 1.0i,
      1.0 + 1.0i, 3.0,
    };
    // clang-format on
    const auto n = kspc::dim(A);
    std::vector<double> w(n);
    std::vector<std::complex<double>> work(4 * n);
    std::vector<double> rwork(3 * n - 2);
    const auto info = kspc::hermitian::eigen_solve(A, w, work, rwork);
    CHECK(info == 0);
    CHECK(equal_to(w[0], 1.0));
    CHECK(equal_to(w[1], 4.0));
  }
  { // hermitian::eigen_solve with row-major dynamic matrix
    using namespace std::complex_literals;
    // clang-format off
    std::vector<std::complex<double>> A{
      2.0, 1.0 + 1.0i,
      1.0 - 1.0i, 3.0,
    };
    // clang-format on
    const auto n = kspc::dim(A);
    std::vector<double> w(n);
    const auto row_major = kspc::mapping::row_major(n);
    const auto info = kspc::hermitian::eigen_solve(A, w, row_major);
    CHECK(info == 0);
    CHECK(equal_to(w[0], 1.0));
    CHECK(equal_to(w[1], 4.0));
  }
  { // hermitian::eigen_solve with row-major static matrix
    using namespace std::complex_literals;
    // clang-format off
    std::array<std::complex<double>, 4> A{
      2.0, 1.0 + 1.0i,
      1.0 - 1.0i, 3.0,
    };
    // clang-format on
    constexpr auto N = kspc::fixed_size_matrix_dim_v<std::remove_cv_t<decltype(A)>>;
    std::array<double, N> w;
    constexpr auto row_major = kspc::mapping::row_major(N);
    const auto info = kspc::hermitian::eigen_solve(A, w, row_major);
    CHECK(info == 0);
    CHECK(equal_to(w[0], 1.0));
    CHECK(equal_to(w[1], 4.0));
  }
}
