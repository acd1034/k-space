#include <iostream>
#include <memory>
#include <kspc/draft.hpp>
#include <kspc/kspc.hpp>

struct X {
  friend inline constexpr std::ptrdiff_t operator-(const X&, const X&) {
    return 0;
  }
};

struct Y {
  using element_type = const int;
  using value_type = int;
};

int main() {
  [[maybe_unused]] kspc::matrix a{1, 2, 3, 4};
  std::cout << std::boolalpha << (1.0 != kspc::approx(1.0 + 2e-6)) << std::endl;
  std::cout << (1.0 == kspc::approx(1.0 + 2e-7)) << std::endl;
  std::cout << (1.0 != kspc::approx(1.0 + 0.20, 0.1, 0.01)) << std::endl;
  std::cout << (1.0 == kspc::approx(1.0 + 0.02, 0.1, 0.01)) << std::endl;
  std::cout << (1.0 != kspc::approx(1.0 + 2.0, 0.1, 1.0)) << std::endl;
  std::cout << (1.0 == kspc::approx(1.0 + 0.2, 0.1, 1.0)) << std::endl;

  std::cout << kspc::is_fixed_size_matrix_v<std::array<int, 3>> << std::endl;
  std::cout << kspc::is_fixed_size_matrix_v<kspc::matrix<int, 3>> << std::endl;
  std::cout << !kspc::is_fixed_size_matrix_v<std::vector<int>> << std::endl;

  // 14, 5, 4, 5, 4, 6, 12
  // clang-format off
  std::cout <<
  kspc::innerp(std::vector{1, 2, 3}, std::vector{1, 2, 3}) << std::endl;
  std::cout <<
  kspc::innerp(std::vector{1, 2}, kspc::matrix{1, 0, 0, 1}, std::vector{1, 2}) << std::endl;
  std::cout <<
  kspc::innerp(std::vector{1, 2}, kspc::matrix{0, 1, 1, 0}, std::vector{1, 2}) << std::endl;
  std::cout <<
  kspc::innerp(std::vector{1, 2}, kspc::ndmatrix{1, 0, 0, 1}, std::vector{1, 2}) << std::endl;
  std::cout <<
  kspc::innerp(std::vector{1, 2}, kspc::ndmatrix{0, 1, 1, 0}, std::vector{1, 2}) << std::endl;
  std::cout <<
  kspc::sum(std::vector{1, 2, 3}) << std::endl;
  std::cout <<
  kspc::sum(std::vector{1, 2, 3}, [](const auto& x) { return 2 * x; }) << std::endl;
  // clang-format on

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
  // error expected
  // static_assert(std::is_same_v<
  //   kspc::incrementable_traits<void*>::difference_type,
  //   std::ptrdiff_t>);

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
