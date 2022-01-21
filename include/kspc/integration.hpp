/// @file integration.hpp
#pragma once
#include <cstdlib> // abort
#include <vector>
#include <gsl/gsl_errno.h> // GSL_ETOL, GSL_EDIVERGE, gsl_stream_printf, gsl_set_error_handler
#include <gsl/gsl_integration.h>
#include <kspc/core.hpp>

namespace kspc {
  /// @addtogroup integration
  /// @{

  /// @brief custom gsl error handler ignoring GSL_ETOL and GSL_EDIVERGE
  /// @details
  /// GSL_ETOL     = 14, failed to reach the specified tolerance
  /// GSL_EDIVERGE = 22, integral or series is divergent
  void error_handler(const char* reason, const char* file, int line, int gsl_errno) {
    if (gsl_errno == GSL_ETOL or gsl_errno == GSL_EDIVERGE) return;
    gsl_stream_printf("ERROR", file, line, reason);
    std::abort();
  }

  /// set custom gsl error handler
  auto set_error_handler() {
    return gsl_set_error_handler(&error_handler);
  }

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

  /// @}
} // namespace kspc

// non-adaptive Gauss-Kronrod integration
namespace kspc::qng {
  /// @addtogroup integration
  /// @{

  /// @cond
  namespace detail {
    /// integrate with gsl_integration_qng
    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params);

    /// integrand for gsl_integration_qng
    template <std::size_t D>
    double integrand(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_x[D] = x;
      return std::get<0>(integrate_impl<D>(void_params));
    }

    /// full specialization of `integrand`
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
      double result, abserr;
      std::size_t nevals;

      // clang-format off
      gsl_integration_qng(&function,
                          params->a[D - 1],
                          params->b[D - 1],
                          params->epsabs,
                          params->epsrel,
                          &result,
                          &abserr,
                          &nevals /* nullptr */);
      // clang-format on

      return {result, abserr};
    }
  } // namespace detail
  /// @endcond

  /// non-adaptive Gauss-Kronrod integration
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

  /// @}
} // namespace kspc::qng

// adaptive integration
namespace kspc::qag {
  /// @addtogroup integration
  /// @{

  /// @cond
  namespace detail {
    /// integrate with gsl_integration_qag
    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params);

    /// integrand for gsl_integration_qag
    template <std::size_t D>
    double integrand(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_x[D] = x;
      return std::get<0>(integrate_impl<D>(void_params));
    }

    /// full specialization of `integrand`
    template <>
    double integrand<0>(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_x[0] = x;
      return (params->workspace_function)(params->workspace_x, void_params);
    }

    inline constexpr int key = 6;

    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params) {
      static_assert(D > 0);
      assert(workspace_size > 1); // `limit` should be larger that 1
      gsl_function function{&integrand<D - 1>, void_params};
      auto* params = (params_t*)void_params;
      auto* workspace = gsl_integration_workspace_alloc(workspace_size);
      double result, abserr;

      // clang-format off
      gsl_integration_qag(&function,
                          params->a[D - 1],
                          params->b[D - 1],
                          params->epsabs,
                          params->epsrel,
                          /* limit */ workspace_size,
                          key,
                          workspace,
                          &result,
                          &abserr);
      // clang-format on

      gsl_integration_workspace_free(workspace);
      return {result, abserr};
    }
  } // namespace detail
  /// @endcond

  /// adaptive integration
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

  /// @}
} // namespace kspc::qag

// doubly-adaptive integration
namespace kspc::cquad {
  /// @addtogroup integration
  /// @{

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

    /// full specialization of `integrand`
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
      std::size_t nevals;

      // clang-format off
      gsl_integration_cquad(&function,
                            params->a[D - 1],
                            params->b[D - 1],
                            params->epsabs,
                            params->epsrel,
                            workspace,
                            &result,
                            &abserr,
                            &nevals);
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

  /// @}
} // namespace kspc::cquad
