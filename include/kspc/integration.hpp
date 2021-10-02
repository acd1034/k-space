/// @file integration.hpp
#pragma once
#include <vector>
#include <gsl/gsl_integration.h>
#include <kspc/math_basics.hpp>

namespace kspc {
  /// helper class for defining parameters of integrand
  struct params_t {
    // double (*p_fn)(const std::vector<double>& x, void* p_params);
    builtin_function<double, const std::vector<double>&, void*>* p_fn;
    std::vector<double> a;
    std::vector<double> b;
    double epsabs;
    double epsrel;
    // workspace
    std::vector<double> tmp_x;
  }; // struct params_t

  /// @cond
  namespace detail {
    inline constexpr std::size_t wssize = 1000;

    template <std::size_t D>
    double cquad_integrand(double x, void* temp_p_params) {
      auto* p_params = (params_t*)temp_p_params;
      p_params->tmp_x[D] = x;
      gsl_function F = {&cquad_integrand<D - 1>, (void*)p_params};
      double result, error;
      std::size_t nevals;

      gsl_integration_cquad_workspace* ws = gsl_integration_cquad_workspace_alloc(wssize);
      // clang-format off
      gsl_integration_cquad(&F,
                            p_params->a[D - 1], p_params->b[D - 1],
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
      p_params->tmp_x[0] = x;
      return (*(p_params->p_fn))(p_params->tmp_x, (void*)p_params);
    }
  } // namespace detail
  /// @endcond

  /// D-dimensional integral
  template <std::size_t D>
  auto cquad(void* temp_p_params) {
    auto* p_params = (params_t*)temp_p_params;
    assert(std::size(p_params->a) >= D);
    assert(std::size(p_params->b) >= D);

    p_params->tmp_x.resize(D);
    gsl_function F = {&detail::cquad_integrand<D - 1>, (void*)p_params};
    double result, error;
    std::size_t nevals;

    gsl_integration_cquad_workspace* ws = gsl_integration_cquad_workspace_alloc(detail::wssize);
    // clang-format off
    gsl_integration_cquad(&F,
                          p_params->a[D - 1], p_params->b[D - 1],
                          p_params->epsabs, p_params->epsrel,
                          ws,
                          &result, &error, &nevals);
    // clang-format on
    gsl_integration_cquad_workspace_free(ws);
    return std::make_pair(result, error);
  }
} // namespace kspc
