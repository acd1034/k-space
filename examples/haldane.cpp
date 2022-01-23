/**
 * @file haldane_fixed.cpp
 * compiler: GCC version 10.2.0
 * compiler option:
 * -O3 -std=c++17 -lm -llapack -lblas -lgsl -lgslcblas -mtune=native -march=native -mfpmath=both
 */
#include <fstream>
#include <iostream>
#include <kspc/integration.hpp>
#include <kspc/linalg.hpp>
#include <kspc/math.hpp>
#include <kspc/numeric.hpp>
using namespace kspc::arithmetic_ops;

inline constexpr std::size_t Nsite = 2;

// nearest neighbor lattice vectors
inline constexpr std::array a{
  std::array{kspc::sqrt3 / 2, 1.0 / 2},
  std::array{-kspc::sqrt3 / 2, 1.0 / 2},
  std::array{0.0, -1.0},
};

// next nearest neighbor lattice vectors
inline constexpr std::array b{
  a[2] - a[1],
  a[0] - a[2],
  a[1] - a[0],
};

// reciprocal lattice vectors
inline constexpr std::array g{
  std::array{-2.0 * kspc::pi / kspc::sqrt3, 2.0 * kspc::pi / 3},
  std::array{-2.0 * kspc::pi / kspc::sqrt3, -2.0 * kspc::pi / 3},
  std::array{0.0, -4.0 * kspc::pi / 3},
};

// parameters of integrand
struct params_t : kspc::params_t {
  double t = 1.0;
  double t2 = 0.0;
  double phi = 0.0;
  double delta = 0.0;
};

// clang-format off

auto gen_hermitian_matrix(const double d0, const double dx, const double dy, const double dz) {
  return std::array<std::complex<double>, Nsite * Nsite>{
    d0 + dz, std::complex{dx, -dy},
    std::complex{dx, dy}, d0 - dz
  };
}

auto H_(const std::vector<double>& k, void* temp_p_params) {
  auto* p = (params_t*)temp_p_params;
  return gen_hermitian_matrix(
    2.0 * p->t2 * std::cos(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return std::cos(kspc::innerp(k, bi)); }),
    p->t
      * kspc::sum(a, [&k](const auto& ai){ return std::cos(kspc::innerp(k, ai)); }),
    p->t
      * kspc::sum(a, [&k](const auto& ai){ return std::sin(kspc::innerp(k, ai)); }),
    p->delta + 2.0 * p->t2 * std::sin(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return std::sin(kspc::innerp(k, bi)); })
  );
}

auto dHdkx_(const std::vector<double>& k, void* temp_p_params) {
  auto* p = (params_t*)temp_p_params;
  return gen_hermitian_matrix(
    -2.0 * p->t2 * std::cos(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return bi[0] * std::sin(kspc::innerp(k, bi)); }),
    -p->t
      * kspc::sum(a, [&k](const auto& ai){ return ai[0] * std::sin(kspc::innerp(k, ai)); }),
    p->t
      * kspc::sum(a, [&k](const auto& ai){ return ai[0] * std::cos(kspc::innerp(k, ai)); }),
    p->delta + 2.0 * p->t2 * std::sin(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return bi[0] * std::cos(kspc::innerp(k, bi)); })
  );
}

auto dHdky_(const std::vector<double>& k, void* temp_p_params) {
  auto* p = (params_t*)temp_p_params;
  return gen_hermitian_matrix(
    -2.0 * p->t2 * std::cos(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return bi[1] * std::sin(kspc::innerp(k, bi)); }),
    -p->t
      * kspc::sum(a, [&k](const auto& ai){ return ai[1] * std::sin(kspc::innerp(k, ai)); }),
    p->t
      * kspc::sum(a, [&k](const auto& ai){ return ai[1] * std::cos(kspc::innerp(k, ai)); }),
    p->delta + 2.0 * p->t2 * std::sin(p->phi)
      * kspc::sum(b, [&k](const auto& bi){ return bi[1] * std::cos(kspc::innerp(k, bi)); })
  );
}

// clang-format on

inline constexpr std::size_t n = 0;
inline constexpr auto in_brillouin_zone = kspc::in_brillouin_zone(g);

// calclate z-component of Berry curvature
double Bz_(const std::vector<double>& k, void* temp_p_params) {
  if (not in_brillouin_zone(k)) return 0.0;

  auto H = H_(k, temp_p_params);
  std::array<double, Nsite> E;
  constexpr auto row_major = kspc::mapping_row_major(Nsite);
  kspc::hermitian::eigen_solve(H, E, row_major);

  auto dHdkx = dHdkx_(k, temp_p_params);
  auto dHdky = dHdky_(k, temp_p_params);
  kspc::unitary_transform(dHdkx, H, row_major, row_major);
  kspc::unitary_transform(dHdky, H, row_major, row_major);

  double bz = 0.0;
  for (std::size_t m = 0; m < Nsite; ++m) {
    if (m != n)
      bz -=
        2.0 * std::imag(dHdkx[row_major(n, m)] * dHdky[row_major(m, n)]) / std::pow(E[n] - E[m], 2);
  }
  return bz;
}

int main() {
  params_t params;
  params.listb = std::vector{kspc::pi, kspc::pi};
  params.lista = -params.listb;
  params.epsabs = 1e-4;
  params.epsrel = 1e-4;
  params.workspace_size = 100;
  params.t2 = 1.0;
  kspc::set_error_handler();
  std::array phi_arr{-kspc::pi / 2, 0.0, kspc::pi / 2};

  for (const auto& phi : phi_arr) {
    params.phi = phi;
    // using kspc::qng::integrate;
    using kspc::qag::integrate;
    // using kspc::cquad::integrate;
    const auto [result, abserr] = integrate<2>(&Bz_, &params);
    std::cout << "phi: " << phi << ", chern #: " << result / 2.0 / kspc::pi << std::endl;
  }
}
