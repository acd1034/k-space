/**
 * @file Haldane.cpp
 * compiler: GCC version 10.2.0
 * compiler option:
 * -O3 -std=c++17 -lm -llapack -lblas -lgsl -lgslcblas -mtune=native -march=native -mfpmath=both
 */
#include <fstream>
#include <iostream>
#include <kspc/integration.hpp>
#include <kspc/kspc.hpp>
#include <kspc/linalg.hpp>
using namespace kspc::arithmetic_ops;

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

// reciprocal lattice vector
inline constexpr std::array g{
  std::array{-2.0 * kspc::pi / kspc::sqrt3, 2.0 * kspc::pi / 3},
  std::array{-2.0 * kspc::pi / kspc::sqrt3, -2.0 * kspc::pi / 3},
  std::array{0.0, -4.0 * kspc::pi / 3},
};

// params of integrand
struct params_t : kspc::params_t {
  double t = 1.0;
  double t2 = 0.0;
  double phi = 0.0;
  double delta = 0.0;
}; // struct params_t

// clang-format off
auto gen_hermitian_matrix(const double d0, const double dx, const double dy, const double dz) {
  return kspc::ndmatrix<std::complex<double>>{
    d0 + dz, std::complex{dx, -dy},
    std::complex{dx, dy}, d0 - dz};
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

inline constexpr std::size_t Nsite = 2;
inline constexpr std::size_t n = 0;
inline constexpr auto in_BZ = kspc::make_in_BZ(g);

double Bz_(const std::vector<double>& k, void* temp_p_params) {
  if (!in_BZ(k)) return 0.0;
  auto H = H_(k, temp_p_params);
  assert(!H.is_hermitian());
  auto [E, U] = kspc::zheev<Nsite>(H);
  auto dHdkx = kspc::mel(dHdkx_(k, temp_p_params), U);
  auto dHdky = kspc::mel(dHdky_(k, temp_p_params), U);

  double bz = 0.0;
  for (std::size_t m = 0; m < Nsite; ++m) {
    if (m != n) bz -= 2.0 * std::imag(dHdkx(n, m) * dHdky(m, n)) / std::pow(E[n] - E[m], 2);
  }
  return bz;
}

int main() {
  params_t params;
  params.p_fn = &Bz_;
  params.b = std::vector{kspc::pi, kspc::pi};
  params.a = -params.b;
  params.epsabs = 1e-4;
  params.epsrel = 1e-4;
  params.t2 = 1.0;
  std::array phi_arr{-kspc::pi / 2, 0.0, kspc::pi / 2};

  for (const auto& phi : phi_arr) {
    params.phi = phi;
    const auto [value, error] = kspc::cquad<Nsite>(&params);
    // clang-format off
    std::cout
      << "phi: " << phi
      << ", chern #: " << value / 2.0 / kspc::pi << std::endl;
    // clang-format on
  }
}
