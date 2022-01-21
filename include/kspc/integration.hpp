/// @file integration.hpp
#pragma once
#include <vector>
#include <gsl/gsl_integration.h>
#include <kspc/math_basics.hpp>

/// @addtogroup integration
/// @{

namespace kspc {
  /// size of workspace
  inline constexpr std::size_t workspace_size = 1000;

  /// type of function to be handled in this library
  using function_t = double(const std::vector<double>&, void*);

  /// helper class to set parameters of integrand
  struct params_t {
    std::vector<double> a;
    std::vector<double> b;
    double epsabs;
    double epsrel;
    // workspace
    function_t* workspace_function;
    std::vector<double> workspace_x;
  }; // struct params_t
} // namespace kspc

// doubly-adaptive integration
namespace kspc::cquad {
  /// @cond
  namespace detail {
    /// integrate with gsl_integration_cquad
    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params);

    /// integrand for gsl_integration_cquad
    template <std::size_t D>
    double integrand(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_x[D] = x;
      return std::get<0>(integrate_impl<D>(void_params));
    }

    /// full specialization of integrand
    template <>
    double integrand<0>(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_x[0] = x;
      return (params->workspace_function)(params->workspace_x, void_params);
    }

    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params) {
      static_assert(D > 0);
      gsl_function function{&integrand<D - 1>, void_params};
      auto* params = (params_t*)void_params;
      auto* workspace = gsl_integration_cquad_workspace_alloc(workspace_size);
      double result, abserr;

      // clang-format off
      gsl_integration_cquad(&function,
                            params->a[D - 1],
                            params->b[D - 1],
                            params->epsabs,
                            params->epsrel,
                            workspace,
                            &result,
                            &abserr,
                            /* &nevals */ nullptr);
      // clang-format on

      gsl_integration_cquad_workspace_free(workspace);
      return {result, abserr};
    }
  } // namespace detail
  /// @endcond

  /// doubly-adaptive integration
  template <std::size_t D>
  auto integrate(function_t* function, void* void_params) {
    static_assert(D > 0);
    auto* params = (params_t*)void_params;
    assert(std::size(params->a) == D);
    assert(std::size(params->b) == D);
    params->workspace_function = function;
    params->workspace_x.resize(D);
    return detail::integrate_impl<D>(void_params);
  }
} // namespace kspc::cquad

/// @}
