/// @file integration.hpp
#pragma once
#include <vector>
#include <gsl/gsl_integration.h>
#include <kspc/math.hpp>

namespace kspc {
  struct params_t {
    // double (*p_fn)(const std::vector<double>& x, void* p_params);
    builtin_function<double, const std::vector<double>&, void*>* p_fn;
    std::vector<double> a;
    std::vector<double> b;
    double epsabs;
    double epsrel;
    // workspace
    std::vector<double> temp_x;
  }; // struct params_t

  namespace detail {
    inline constexpr std::size_t wssize = 1000;

    template <std::size_t N>
    double cquad_integrand(double x, void* temp_p_params) {
      auto* p_params = (params_t*)temp_p_params;
      p_params->temp_x[N] = x;
      gsl_function F = {&cquad_integrand<N - 1>, (void*)p_params};
      double result, error;
      std::size_t nevals;

      gsl_integration_cquad_workspace* ws = gsl_integration_cquad_workspace_alloc(wssize);
      // clang-format off
      gsl_integration_cquad(&F,
                            p_params->a[N - 1], p_params->b[N - 1],
                            p_params->epsabs, p_params->epsrel,
                            ws,
                            &result, &error, &nevals);
      // clang-format on
      gsl_integration_cquad_workspace_free(ws);
      return result;
    }

    template <>
    double cquad_integrand<0>(double x, void* temp_p_params) {
      auto* p_params = (params_t*)temp_p_params;
      p_params->temp_x[0] = x;
      return (*(p_params->p_fn))(p_params->temp_x, (void*)p_params);
    }
  } // namespace detail

  template <std::size_t N>
  auto cquad(void* temp_p_params) {
    auto* p_params = (params_t*)temp_p_params;
    assert(std::size(p_params->a) >= N);
    assert(std::size(p_params->b) >= N);

    p_params->temp_x.resize(N);
    gsl_function F = {&detail::cquad_integrand<N - 1>, (void*)p_params};
    double result, error;
    std::size_t nevals;

    gsl_integration_cquad_workspace* ws = gsl_integration_cquad_workspace_alloc(detail::wssize);
    // clang-format off
    gsl_integration_cquad(&F,
                          p_params->a[N - 1], p_params->b[N - 1],
                          p_params->epsabs, p_params->epsrel,
                          ws,
                          &result, &error, &nevals);
    // clang-format on
    gsl_integration_cquad_workspace_free(ws);
    return std::make_pair(result, error);
  }
} // namespace kspc
