#include <cmath>
#include <cstdio>
#include <kspc/integration.hpp>

struct params_t : kspc::params_t {
  double alpha;
};

double f(const std::vector<double>& x, void* void_params) {
  auto* params = (params_t*)void_params;
  return std::log(params->alpha * x[0]) / std::sqrt(x[0]);
}

int main() {
  params_t params;
  params.a = std::vector{0.0};
  params.b = std::vector{1.0};
  params.epsabs = 0.0;
  params.epsrel = 1e-7;
  params.alpha = 1.0;

  using kspc::cquad::integrate;
  const auto [result, abserr] = integrate<1>(&f, &params);
  const double expected = -4.0;

  printf("result          = % .18f\n", result);
  printf("exact result    = % .18f\n", expected);
  printf("estimated error = % .18f\n", abserr);
  printf("actual error    = % .18f\n", result - expected);
}
