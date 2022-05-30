/// @file iso2d.hpp
#pragma once
#include <array>
#include <cmath>    // std::hypot
#include <iterator> // std::size
#include <utility>
#include <vector>
#include <kspc/linalg.hpp> // kspc::mapping::row_major
#include <kspc/core.hpp> // not used

namespace iso2d {
  //// parameters to generate cartesian grid
  struct CartesianGrid {
    std::size_t nx, ny;
    double x, dx, y, dy;
  };

  /// type of function to be handled in this library
  using function_t = double(const std::array<double, 2>&, void*);

  /// helper class to set function to extract isoline
  struct params_t {
    function_t* function;
    double iso = 0.0;
  };

  namespace detail {
    double function_call(const std::array<double, 2>& v, void* data) {
      auto* params = (params_t*)data;
      return (params->function)(v, data) - params->iso;
    }

    struct BSearchForRootFn {
      std::array<double, 2> v;
      const std::size_t n;
      void* data;
      BSearchForRootFn(double x, std::size_t m, void* d) : n(m < 1 ? m : 1), data(d) {
        v[1 - n] = x;
      }
      double operator()(double y) {
        v[n] = y;
        return function_call(v, data);
      }
    };

    bool have_opposite_signs(double v1, double v2) {
      return (v1 >= 0. and v2 < 0.) or (v1 < 0. and v2 >= 0.);
    }

    double internal_div(double x1, double x2, double v1, double v2) {
      double c = v1 / (v1 - v2);
      if (have_opposite_signs(x1, x2)) return c * x2 + (1. - c) * x1;
      return x1 + c * (x2 - x1);
    }

    double bsearch_for_root(double x1, double x2, double v1, double v2, BSearchForRootFn&& fn) {
      double xmid, vmid;
      while (x2 - x1 > 1e-6) {
        xmid = (x1 + x2) / 2.; // Optimized for small case
        vmid = fn(xmid);
        if (have_opposite_signs(vmid, v2))
          x1 = xmid, v1 = vmid;
        else {
          assert(have_opposite_signs(vmid, v1));
          x2 = xmid, v2 = vmid;
        }
      }
      return internal_div(x1, x2, v1, v2);
    }

    auto isoline_cartesian_impl(CartesianGrid g, void* data) {
      assert(g.nx > 1);
      assert(g.ny > 1);

      std::size_t i, j;
      std::array<double, 2> v;
      std::vector<double> f(g.nx * g.ny);
      // row-major: i < nx, j < ny → A(i, j) = A[i * ny + j]
      const kspc::mapping::row_major at(g.ny);
      for (i = 0, v[0] = g.x; i < g.nx; ++i, v[0] += g.dx)
        for (j = 0, v[1] = g.y; j < g.ny; ++j, v[1] += g.dy) at(f, i, j) = function_call(v, data);

      std::vector<std::array<double, 2>> vertices;
      vertices.reserve(2 * g.nx * g.ny);
      // sweep along x-axis
      for (i = 0, v[0] = g.x; i < g.nx - 1; ++i, v[0] += g.dx)
        for (j = 0, v[1] = g.y; j < g.ny; ++j, v[1] += g.dy) {
          double v1 = at(f, i, j);
          double v2 = at(f, i + 1, j);
          if (have_opposite_signs(v1, v2)) {
            double x = bsearch_for_root(v[0], v[0] + g.dx, v1, v2, BSearchForRootFn(v[1], 0, data));
            vertices.push_back({x, v[1]});
          }
        }
      // sweep along y-axis
      for (i = 0, v[0] = g.x; i < g.nx; ++i, v[0] += g.dx)
        for (j = 0, v[1] = g.y; j < g.ny - 1; ++j, v[1] += g.dy) {
          double v1 = at(f, i, j);
          double v2 = at(f, i, j + 1);
          if (have_opposite_signs(v1, v2)) {
            double y = bsearch_for_root(v[1], v[1] + g.dy, v1, v2, BSearchForRootFn(v[0], 1, data));
            vertices.push_back({v[0], y});
          }
        }
      vertices.shrink_to_fit();

      std::vector<std::array<std::size_t, 2>> lines;
      lines.reserve(16 * std::size(vertices));
      const double d = std::hypot(g.dx, g.dy);
      for (i = 0; i < std::size(vertices) - 1; ++i)
        for (j = i + 1; j < std::size(vertices); ++j) {
          if (std::hypot(vertices[j][0] - vertices[i][0], vertices[j][1] - vertices[i][1]) < d)
            lines.push_back({i, j});
        }
      lines.shrink_to_fit();

      return std::make_pair(std::move(vertices), std::move(lines));
    }
  } // namespace detail

  /// extract isoline
  auto isoline_cartesian(CartesianGrid grid, function_t* fn, void* data, double iso) {
    auto* params = (params_t*)data;
    params->function = fn;
    params->iso = iso;
    return detail::isoline_cartesian_impl(grid, data);
  }
} // namespace iso2d
