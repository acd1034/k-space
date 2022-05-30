/**
 * @file primitive-cubic.cpp
 * @brief Cross section of Fermi surface at k_z = 0
 */
#include <array>
#include <cmath>
#include <iostream>
#include <kspc/iso2d.hpp> // isoline_cartesian
namespace iso2d = kspc::iso2d;

// lattice constant
inline constexpr double la = 1.0;

struct params_t : iso2d::params_t {
  double t = 1.0;
  double mu = 0.0;
};

double E(const std::array<double, 3>& k, void* data) {
  const auto& p = *(params_t*)data;
  return -2.0 * p.t * (std::cos(k[0] * la) + std::cos(k[1] * la) + std::cos(k[2] * la));
}

double fn1(const std::array<double, 2>& k, void* data) {
  // const auto& p = *(params_t*)data;
  std::array<double, 3> k3d{k[0], k[1], 0.0};
  return E(k3d, data);
}

double fn2(const std::array<double, 2>& k, void* /* data */) {
  // const auto& p = *(params_t*)data;
  return std::abs(k[0]) < 1e-6 ? 0.0 : std::sin(4.0 * std::atan2(k[1], k[0]));
}

int main() {
  params_t params;
  // params.mu = 0.0;
  constexpr double k0 = kspc::pi / la;
  auto grid = iso2d::symmetric_grid(-k0, k0, -k0, k0, 100);
  const auto& [vertices, lines] = iso2d::isoline_cartesian(grid, &fn1, &params, params.mu);

  std::cout << "# " << std::size(vertices) << " " << std::size(lines) << "\n";
  for (const auto& v : vertices) {
    const auto sz = fn2(v, &params);
    std::cout << v[0] << " " << v[1] << " " << sz << "\n";
  }
  for (const auto& [x, y] : lines) std::cout << x << " " << y << "\n";
}
