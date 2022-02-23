#include <cmath>
#include <cstdio>
#include <kspc/integration.hpp>
#include <kspc/linalg.hpp>
#include <kspc/math.hpp>
#include <kspc/numeric.hpp>
using namespace kspc::arithmetic_ops;

// lattice constant
inline constexpr double la = 1.0;

struct params_t : kspc::params_t {
  double t = 1.0;
  double mu;
};

double E(const std::vector<double>& k, void* void_params) {
  const auto& p = *(params_t*)void_params;
  return -2.0 * p.t * (std::cos(k[0] * la) + std::cos(k[1] * la) + std::cos(k[2] * la));
}

double Ez(const std::vector<double>& k, void* void_params) {
  const auto& p = *(params_t*)void_params;
  return 2.0 * p.t * la * std::sin(k[2] * la);
}

double Ezz(const std::vector<double>& k, void* void_params) {
  const auto& p = *(params_t*)void_params;
  return 2.0 * p.t * la * la * std::cos(k[2] * la);
}

double f(const std::vector<double>& k, void* void_params) {
  const auto& p = *(params_t*)void_params;
  if (E(k, void_params) > p.mu) return 0.0;
  return Ezz(k, void_params);
}

void error_handler(const char* reason, const char* file, int line, int gsl_errno) {
  if (gsl_errno == GSL_EMAXITER or gsl_errno == GSL_ETOL or gsl_errno == GSL_EROUND) return;
  gsl_stream_printf("ERROR", file, line, reason);
  std::abort();
}

int main() {
  params_t params;
  params.listb = std::vector{kspc::pi / la, kspc::pi / la, kspc::pi / la};
  params.lista = -params.listb;
  params.epsabs = 1e-6;
  params.epsrel = 1e-6;
  params.workspace_size = 100;
  // params.mu = 0.0;
  gsl_set_error_handler(&error_handler);

  for (const auto& mu : std::array{0.0, 2.0, 4.0}) {
    params.mu = mu;
    // using kspc::qng::integrate;
    using kspc::qag::integrate;
    // using kspc::cquad::integrate;
    const auto [result, abserr, info] = integrate<3>(&f, &params);

    printf("result          = % .6f\n", result);
    printf("estimated error = % .6f\n", abserr);
  }
}

// result          =  82.883472
// estimated error =   0.000001
// result          =  59.125300
// estimated error =   0.000001
// result          =  22.348181
// estimated error =   0.000011
