#define CATCH_CONFIG_MAIN
#include <iostream>
#include <memory>
#include <catch2/catch.hpp>
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
