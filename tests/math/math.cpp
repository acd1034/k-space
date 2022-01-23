#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm> // equal
#include <array>
#include <complex>
#include <memory> // shared_ptr
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

struct X {};

inline constexpr std::ptrdiff_t operator-(const X&, const X&) {
  return 0;
}

struct Y {
  using element_type = const int;
  using value_type = int;
};

TEST_CASE("core", "[core]") {
  // clang-format off
  // incrementable_traits
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

  // indirectly_readable_traits
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
  { // dim
    std::array<int, 4> a;
    static_assert(kspc::fixed_size_matrix_dim_v<std::remove_cv_t<decltype(a)>> == 2);
    std::vector<int> v(4);
    CHECK(kspc::dim(v) == 2);
  }
}

TEST_CASE("mapping", "[math][mapping]") {
  { // mapping_row_major
    constexpr auto mapping = kspc::mapping_row_major(5);
    CHECK(mapping(1, 2) == 7);
  }
  { // mapping_column_major
    constexpr auto mapping = kspc::mapping_column_major(5);
    CHECK(mapping(1, 2) == 11);
  }
}

TEST_CASE("projection", "[math][projection]") {
  { // identity
    std::complex c{1.0, 1.0};
    CHECK(equal_to(kspc::identity(c), std::complex{1.0, 1.0}));
    CHECK(equal_to(kspc::identity(std::complex{1.0, 1.0}), std::complex{1.0, 1.0}));
    double d = 1.0;
    CHECK(equal_to(kspc::identity(d), 1.0));
    CHECK(equal_to(kspc::identity(1.0), 1.0));
  }
  // is_complex_v
  static_assert(kspc::is_complex_v<std::complex<double>>);
  static_assert(not kspc::is_complex_v<double>);
  // complex_value_t
  static_assert(std::is_same_v<kspc::complex_value_t<std::complex<double>>, double>);
  static_assert(std::is_same_v<kspc::complex_value_t<const std::complex<double>>, double>);
  static_assert(std::is_same_v<kspc::complex_value_t<const std::complex<double>&>, double>);
  static_assert(std::is_same_v<kspc::complex_value_t<double>, double>);
  static_assert(std::is_same_v<kspc::complex_value_t<const double>, double>);
  static_assert(std::is_same_v<kspc::complex_value_t<const double&>, double>);
  { // conj
    std::complex c{1.0, 1.0};
    CHECK(equal_to(kspc::conj(c), std::complex{1.0, -1.0}));
    CHECK(equal_to(kspc::conj(std::complex{1.0, 1.0}), std::complex{1.0, -1.0}));
    double d = 1.0;
    CHECK(equal_to(kspc::conj(d), 1.0));
    CHECK(equal_to(kspc::conj(1.0), 1.0));
  }
}

TEST_CASE("numeric", "[numeric]") {
  { // sum
    constexpr auto twice = [](const auto& x) { return 2 * x; };
    const std::vector v{1, 2, 3};
    CHECK(kspc::sum(v) == 6);
    CHECK(kspc::sum(v, twice) == 12);
  }
  { // innerp
    using namespace std::complex_literals;
    const std::vector v{1.0, 2.0, 3.0};
    const std::vector vc{1.0i, 2.0i, 3.0i};
    CHECK(equal_to(kspc::innerp(v, v), 14.0));
    CHECK(equal_to(kspc::innerp(vc, vc), 14.0));
  }
  // { // innerp with op
  //   CHECK(kspc::innerp(std::vector{1, 2}, std::vector{1, 0, 0, 1}, std::vector{1, 2}) == 5);
  //   CHECK(kspc::innerp(std::vector{1, 2}, std::vector{0, 1, 1, 0}, std::vector{1, 2}) == 4);
  //   CHECK(kspc::innerp(std::vector{1, 2}, std::array{1, 0, 0, 1}, std::vector{1, 2}) == 5);
  //   CHECK(kspc::innerp(std::vector{1, 2}, std::array{0, 1, 1, 0}, std::vector{1, 2}) == 4);
  // }
  { // arithmetic ops with std::array
    using namespace kspc::arithmetic_ops;
    constexpr std::array a1{1, 2, 3};
    constexpr std::array a2{2, 4, 6};
    CHECK(equal(+a1, std::array{1, 2, 3}));
    CHECK(equal(-a1, std::array{-1, -2, -3}));
    CHECK(equal(a1 + a2, std::array{3, 6, 9}));
    CHECK(equal(a2 - a1, std::array{1, 2, 3}));
    CHECK(equal(2 * a1, std::array{2, 4, 6}));
    CHECK(equal(a1 * 2, std::array{2, 4, 6}));
    CHECK(equal(a2 / 2, std::array{1, 2, 3}));
    CHECK(equal(+std::array{1, 2, 3}, std::array{1, 2, 3}));
    CHECK(equal(-std::array{1, 2, 3}, std::array{-1, -2, -3}));
    CHECK(equal(std::array{1, 2, 3} + a2, std::array{3, 6, 9}));
    CHECK(equal(std::array{2, 4, 6} - std::array{1, 2, 3}, std::array{1, 2, 3}));
    CHECK(equal(2 * std::array{1, 2, 3}, std::array{2, 4, 6}));
    CHECK(equal(std::array{1, 2, 3} * 2, std::array{2, 4, 6}));
    CHECK(equal(std::array{2, 4, 6} / 2, std::array{1, 2, 3}));
  }
}

TEST_CASE("approx", "[math][approx]") {
  namespace app = kspc::approx;
  constexpr double eps = 1e-6;
  // clang-format off
  { // approximate comparison for double
    CHECK(             app::less(1.0, 1.0 + 2e-6, eps));
    CHECK(      not app::greater(1.0, 1.0 + 2e-6, eps));
    CHECK(       app::less_equal(1.0, 1.0 + 2e-6, eps));
    CHECK(not app::greater_equal(1.0, 1.0 + 2e-6, eps));
    CHECK(     app::not_equal_to(1.0, 1.0 + 2e-6, eps));
    CHECK(     not app::equal_to(1.0, 1.0 + 2e-6, eps));

    CHECK(        not app::less(1.0, 1.0 + 2e-7, eps));
    CHECK(     not app::greater(1.0, 1.0 + 2e-7, eps));
    CHECK(      app::less_equal(1.0, 1.0 + 2e-7, eps));
    CHECK(   app::greater_equal(1.0, 1.0 + 2e-7, eps));
    CHECK(not app::not_equal_to(1.0, 1.0 + 2e-7, eps));
    CHECK(        app::equal_to(1.0, 1.0 + 2e-7, eps));
  }
  { // approximate comparison for complex
    using namespace std::complex_literals;
    const std::complex<double> c{1.0, 1.0};
    CHECK(app::not_equal_to(c, c + 2e-6, eps));
    CHECK(app::not_equal_to(c, c + 2e-6i, eps));
    CHECK(app::not_equal_to(c, c + 2e-6*c, eps));
    CHECK(not app::equal_to(c, c + 2e-6, eps));
    CHECK(not app::equal_to(c, c + 2e-6i, eps));
    CHECK(not app::equal_to(c, c + 2e-6*c, eps));

    CHECK(not app::not_equal_to(c, c + 2e-7, eps));
    CHECK(not app::not_equal_to(c, c + 2e-7i, eps));
    CHECK(not app::not_equal_to(c, c + 2e-7*c, eps));
    CHECK(        app::equal_to(c, c + 2e-7, eps));
    CHECK(        app::equal_to(c, c + 2e-7i, eps));
    CHECK(        app::equal_to(c, c + 2e-7*c, eps));
  }
  // clang-format on
}

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
    const auto row_major = kspc::mapping_row_major(kspc::dim(A));
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
    constexpr auto row_major = kspc::mapping_row_major(N);
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
    const auto row_major = kspc::mapping_row_major(kspc::dim(A));
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
    constexpr auto row_major = kspc::mapping_row_major(N);
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
    const auto row_major = kspc::mapping_row_major(n);
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
    constexpr auto row_major = kspc::mapping_row_major(N);
    const auto info = kspc::hermitian::eigen_solve(A, w, row_major);
    CHECK(info == 0);
    CHECK(equal_to(w[0], 1.0));
    CHECK(equal_to(w[1], 4.0));
  }
}
