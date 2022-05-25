#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <array>
#include <ranges>
#include <string>
#include <kspc/approx.hpp>
#include <kspc/ranges.hpp>

inline constexpr auto equal_to = [](const auto& x, const auto& y) {
  return kspc::approx::equal_to(x, y, 1e-6);
};

constexpr auto arrange(double start, double bound, double step) {
  if ((start < bound) != (step > 0)) step = -step;
  auto n = static_cast<std::ptrdiff_t>((bound - start) / step) + 1;
  return kspc::kappa_view(start, step, n);
}

TEST_CASE("ranges", "[ranges][kappa_view]") {
  {
    using KV = kspc::kappa_view<double, double>;
    static_assert(std::ranges::forward_range<KV>);
    static_assert(std::ranges::bidirectional_range<KV>);
  }
  {
    using KV = kspc::kappa_view<std::string, std::string>;
    static_assert(std::ranges::forward_range<KV>);
    static_assert(not std::ranges::bidirectional_range<KV>);
  }
  {
    kspc::kappa_view kv(0.0, 1.5, 4);
    std::array a{0.0, 1.5, 3.0, 4.5};
    CHECK(std::ranges::ssize(kv) == std::ranges::ssize(a));
    auto it = std::ranges::begin(kv);
    for (const auto x : a) CHECK(equal_to(*it++, x));
    CHECK(it == std::ranges::end(kv));
  }
  {
    auto kv = arrange(1.2, 5.0, 1.2);
    std::array a{1.2, 2.4, 3.6, 4.8};
    CHECK(std::ranges::ssize(kv) == std::ranges::ssize(a));
    auto it = std::ranges::begin(kv);
    for (const auto x : a) CHECK(equal_to(*it++, x));
    CHECK(it == std::ranges::end(kv));
  }
}
