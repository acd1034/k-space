#define CATCH_CONFIG_MAIN
#include <iostream>
#include <memory>
#include <catch2/catch.hpp>
#include <kspc/linalg.hpp>
#include <kspc/math.hpp>

struct X {};

inline constexpr std::ptrdiff_t operator-(const X&, const X&) {
  return 0;
}

struct Y {
  using element_type = const int;
  using value_type = int;
};

TEST_CASE("control", "[control]") {
  // CHECK(1.0 != kspc::approx(1.0 + 2e-6));
  // CHECK(1.0 == kspc::approx(1.0 + 2e-7));
  // CHECK(1.0 != kspc::approx(1.0 + 0.20, 0.1, 0.01));
  // CHECK(1.0 == kspc::approx(1.0 + 0.02, 0.1, 0.01));
  // CHECK(1.0 != kspc::approx(1.0 + 2.0, 0.1, 1.0));
  // CHECK(1.0 == kspc::approx(1.0 + 0.2, 0.1, 1.0));

  CHECK(kspc::innerp(std::vector{1, 2, 3}, std::vector{1, 2, 3}) == 14);
  CHECK(kspc::innerp(std::vector{1, 2}, std::vector{1, 0, 0, 1}, std::vector{1, 2}) == 5);
  CHECK(kspc::innerp(std::vector{1, 2}, std::vector{0, 1, 1, 0}, std::vector{1, 2}) == 4);
  CHECK(kspc::innerp(std::vector{1, 2}, std::array{1, 0, 0, 1}, std::vector{1, 2}) == 5);
  CHECK(kspc::innerp(std::vector{1, 2}, std::array{0, 1, 1, 0}, std::vector{1, 2}) == 4);
  CHECK(kspc::sum(std::vector{1, 2, 3}) == 6);
  CHECK(kspc::sum(std::vector{1, 2, 3}, [](const auto& x) { return 2 * x; }) == 12);

  // clang-format off
  static_assert(std::is_same_v<
    kspc::incrementable_traits<int*>::difference_type,
    std::ptrdiff_t>);
  static_assert(std::is_same_v<
    kspc::incrementable_traits<std::vector<int>::iterator>::difference_type,
    std::ptrdiff_t>);
  static_assert(std::is_same_v<
    kspc::incrementable_traits<std::vector<int>::reverse_iterator>::difference_type,
    std::ptrdiff_t>);
  static_assert(std::is_same_v<
    kspc::incrementable_traits<const std::vector<int>::iterator>::difference_type,
    std::ptrdiff_t>);
  static_assert(std::is_same_v<
    kspc::incrementable_traits<X>::difference_type,
    std::ptrdiff_t>);

  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<int*>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<int[]>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<int[42]>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<std::vector<int>::iterator>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<std::vector<int>::reverse_iterator>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<std::shared_ptr<int>>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<Y>::value_type,
    int>);
  static_assert(std::is_same_v<
    kspc::indirectly_readable_traits<const std::vector<int>::iterator>::value_type,
    int>);
  // clang-format on
}

TEST_CASE("dim", "[math][dim]") {
  // fixed_size_array_size_v
  static_assert(kspc::fixed_size_array_size_v<int[3]> == 3);
  static_assert(kspc::fixed_size_array_size_v<std::array<int, 3>> == 3);
  // is_fixed_size_array_v
  static_assert(kspc::is_fixed_size_array_v<int[3]>);
  static_assert(kspc::is_fixed_size_array_v<std::array<int, 3>>);
  static_assert(not kspc::is_fixed_size_array_v<std::vector<int>>);
  {
    // dim
    std::array<int, 4> a;
    static_assert(kspc::dim(a) == 2);
    std::vector<int> v(4);
    CHECK(kspc::dim(v) == 2);
  }
}

TEST_CASE("mapping", "[math][mapping]") {
  {
    // mapping_row_major
    constexpr auto mapping = kspc::mapping_row_major(5);
    CHECK(mapping(1, 2) == 7);
  }
  {
    // mapping_column_major
    constexpr auto mapping = kspc::mapping_column_major(5);
    CHECK(mapping(1, 2) == 11);
  }
}

TEST_CASE("projection", "[math][projection]") {
  {
    // identity
    std::complex c{1.0, 1.0};
    CHECK(kspc::identity(c) == std::complex{1.0, 1.0});
    CHECK(kspc::identity(std::complex{1.0, 1.0}) == std::complex{1.0, 1.0});
    double d = 1.0;
    CHECK(kspc::identity(d) == 1.0);
    CHECK(kspc::identity(1.0) == 1.0);
  }
  // is_complex_v
  static_assert(kspc::is_complex_v<std::complex<double>>);
  static_assert(not kspc::is_complex_v<double>);
  {
    // conj
    std::complex c{1.0, 1.0};
    CHECK(kspc::conj(c) == std::complex{1.0, -1.0});
    CHECK(kspc::conj(std::complex{1.0, 1.0}) == std::complex{1.0, -1.0});
    double d = 1.0;
    CHECK(kspc::conj(d) == 1.0);
    CHECK(kspc::conj(1.0) == 1.0);
  }
}

TEST_CASE("matrix", "[math][matrix]") {
  {
    // matrix_vector_solve with column-major dynamic matrix
    // clang-format off
    std::vector A{
       2.0,  2.0,  1.0,
       1.0, -1.0, -1.0,
      -3.0, -1.0, -2.0,
    }; // NOTE: A is column-major
    // clang-format on
    std::vector<std::size_t> ipiv(3);
    std::vector b{-2.0, -2.0, -5.0};
    const auto info = kspc::matrix_vector_solve(A, ipiv, b);
    CHECK(info == 0);
    CHECK(b == std::vector{1.0, 2.0, 2.0});
  }
  {
    // matrix_vector_solve with row-major dynamic matrix
    // clang-format off
    std::vector A{
      2.0,  1.0, -3.0,
      2.0, -1.0, -1.0,
      1.0, -1.0, -2.0,
    };
    // clang-format on
    std::vector b{-2.0, -2.0, -5.0};
    const auto row_major = kspc::mapping_row_major(kspc::dim(A));
    const auto info = kspc::matrix_vector_solve(A, b, row_major);
    CHECK(info == 0);
    CHECK(b == std::vector{1.0, 2.0, 2.0});
  }
  {
    // matrix_vector_solve with row-major static matrix
    // clang-format off
    std::array A{
      2.0,  1.0, -3.0,
      2.0, -1.0, -1.0,
      1.0, -1.0, -2.0,
    };
    // clang-format on
    std::array b{-2.0, -2.0, -5.0};
    constexpr auto row_major = kspc::mapping_row_major(kspc::dim(A));
    const auto info = kspc::matrix_vector_solve(A, b, row_major);
    CHECK(info == 0);
    CHECK(b == std::array{1.0, 2.0, 2.0});
  }
  {
    // hermitian_matrix_eigen_solve with column-major dynamic matrix
    using namespace std::complex_literals;
    // clang-format off
    std::vector<std::complex<double>> A{
      2.0, 1.0 - 1.0i,
      1.0 + 1.0i, 3.0,
    }; // NOTE: A is column-major
    // clang-format on
    const auto n = kspc::dim(A);
    std::vector<double> w(n);
    std::vector<std::complex<double>> work(4 * n);
    std::vector<double> rwork(3 * n - 2);
    const auto info = kspc::hermitian_matrix_eigen_solve(A, w, work, rwork);
    CHECK(info == 0);
    CHECK(w[0] == 1.0);
    CHECK(w[1] == 4.0);
  }
  {
    // hermitian_matrix_eigen_solve with row-major dynamic matrix
    using namespace std::complex_literals;
    // clang-format off
    std::vector<std::complex<double>> A{
      2.0, 1.0 + 1.0i,
      1.0 - 1.0i, 3.0,
    };
    // clang-format on
    const auto n = kspc::dim(A);
    std::vector<double> w(n);
    const auto row_major = kspc::mapping_row_major(n);
    const auto info = kspc::hermitian_matrix_eigen_solve(A, w, row_major);
    CHECK(info == 0);
    CHECK(w[0] == 1.0);
    CHECK(w[1] == 4.0);
  }
  {
    // hermitian_matrix_eigen_solve with row-major static matrix
    using namespace std::complex_literals;
    // clang-format off
    std::array<std::complex<double>, 4> A{
      2.0, 1.0 + 1.0i,
      1.0 - 1.0i, 3.0,
    };
    // clang-format on
    constexpr auto N = kspc::dim(A);
    std::array<double, N> w;
    constexpr auto row_major = kspc::mapping_row_major(N);
    const auto info = kspc::hermitian_matrix_eigen_solve(A, w, row_major);
    CHECK(info == 0);
    CHECK(w[0] == 1.0);
    CHECK(w[1] == 4.0);
  }
}
