/// @file integration.hpp
#pragma once
#include <array>
#include <cstdlib> // abort
#include <vector>
#include <gsl/gsl_errno.h> // GSL_EMAXITER, GSL_ETOL, gsl_stream_printf, gsl_set_error_handler
#include <gsl/gsl_integration.h>
#include <kspc/core.hpp>

namespace kspc {
  /// @addtogroup integration
  /// @{

  /// @brief custom gsl error handler ignoring GSL_EMAXITER and GSL_ETOL
  /// @details
  /// GSL_EMAXITER = 11, exceeded max number of iterations
  /// GSL_ETOL     = 14, failed to reach the specified tolerance
  void error_handler(const char* reason, const char* file, int line, int gsl_errno) {
    if (gsl_errno == GSL_EMAXITER or gsl_errno == GSL_ETOL) return;
    gsl_stream_printf("ERROR", file, line, reason);
    std::abort();
  }

  /// set custom gsl error handler
  auto set_error_handler() {
    return gsl_set_error_handler(&error_handler);
  }

  /// type of function to be handled in this library
  using function_t = double(const std::vector<double>&, void*);

  /// helper class to set parameters of integrand
  struct params_t {
    std::vector<double> lista;
    std::vector<double> listb;
    double epsabs;
    double epsrel;
    std::size_t workspace_size = 1000;
    // workspace
    function_t* workspace_function;
    void* workspace; // instance is `gsl_integration_workspace**` etc.
    std::vector<double> workspace_listx;
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
      params->workspace_listx[D] = x;
      return std::get<0>(integrate_impl<D - 1>(void_params));
    }

    /// full specialization of `integrand`
    template <>
    double integrand<0>(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_listx[0] = x;
      return (params->workspace_function)(params->workspace_listx, void_params);
    }

    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params) {
      gsl_function function{&integrand<D>, void_params};
      auto* params = (params_t*)void_params;
      double result, abserr;
      std::size_t nevals;

      // clang-format off
      gsl_integration_qng(&function,
                          params->lista[D],
                          params->listb[D],
                          params->epsabs,
                          params->epsrel,
                          &result,
                          &abserr,
                          &nevals);
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
    assert(std::size(params->lista) == D);
    assert(std::size(params->listb) == D);

    params->workspace_function = function;
    params->workspace_listx.resize(D);

    return detail::integrate_impl<D - 1>(void_params);
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
      params->workspace_listx[D] = x;
      return std::get<0>(integrate_impl<D - 1>(void_params));
    }

    /// full specialization of `integrand`
    template <>
    double integrand<0>(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_listx[0] = x;
      return (params->workspace_function)(params->workspace_listx, void_params);
    }

    inline constexpr int key = 6;

    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params) {
      gsl_function function{&integrand<D>, void_params};
      auto* params = (params_t*)void_params;
      assert(params->workspace_size > 1);
      auto** ws = (gsl_integration_workspace**)params->workspace;
      double result, abserr;

      // clang-format off
      gsl_integration_qag(&function,
                          params->lista[D],
                          params->listb[D],
                          params->epsabs,
                          params->epsrel,
                          params->workspace_size,
                          key,
                          ws[D],
                          &result,
                          &abserr);
      // clang-format on

      return {result, abserr};
    }
  } // namespace detail
  /// @endcond

  /// adaptive integration
  template <std::size_t D>
  auto integrate(function_t* function, void* void_params) {
    static_assert(D > 0);
    auto* params = (params_t*)void_params;
    assert(std::size(params->lista) == D);
    assert(std::size(params->listb) == D);

    params->workspace_function = function;
    std::array<gsl_integration_workspace*, D> workspace{};
    for (auto& w : workspace) w = gsl_integration_workspace_alloc(params->workspace_size);
    params->workspace = std::data(workspace);
    params->workspace_listx.resize(D);

    auto ret = detail::integrate_impl<D - 1>(void_params);
    for (auto& w : workspace) gsl_integration_workspace_free(w);
    return ret;
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
      params->workspace_listx[D] = x;
      return std::get<0>(integrate_impl<D - 1>(void_params));
    }

    /// full specialization of `integrand`
    template <>
    double integrand<0>(double x, void* void_params) {
      auto* params = (params_t*)void_params;
      params->workspace_listx[0] = x;
      return (params->workspace_function)(params->workspace_listx, void_params);
    }

    template <std::size_t D>
    std::pair<double, double> integrate_impl(void* void_params) {
      gsl_function function{&integrand<D>, void_params};
      auto* params = (params_t*)void_params;
      auto** ws = (gsl_integration_cquad_workspace**)params->workspace;
      double result, abserr;
      std::size_t nevals;

      // clang-format off
      gsl_integration_cquad(&function,
                            params->lista[D],
                            params->listb[D],
                            params->epsabs,
                            params->epsrel,
                            ws[D],
                            &result,
                            &abserr,
                            &nevals);
      // clang-format on

      return {result, abserr};
    }
  } // namespace detail
  /// @endcond

  /// doubly-adaptive integration
  template <std::size_t D>
  auto integrate(function_t* function, void* void_params) {
    static_assert(D > 0);
    auto* params = (params_t*)void_params;
    assert(std::size(params->lista) == D);
    assert(std::size(params->listb) == D);

    params->workspace_function = function;
    std::array<gsl_integration_cquad_workspace*, D> workspace{};
    for (auto& w : workspace) w = gsl_integration_cquad_workspace_alloc(params->workspace_size);
    params->workspace = std::data(workspace);
    params->workspace_listx.resize(D);

    auto ret = detail::integrate_impl<D - 1>(void_params);
    for (auto& w : workspace) gsl_integration_cquad_workspace_free(w);
    return ret;
  }

  /// @}
} // namespace kspc::cquad
