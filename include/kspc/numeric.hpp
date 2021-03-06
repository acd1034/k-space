/// @file numeric.hpp
#pragma once
#include <algorithm> // min
#include <array>
#include <vector>
#include <kspc/core.hpp> // is_input_range, identity_fn, conj_fn

// constexpr invoke (invoke is constexpr after C++20)
// https://en.cppreference.com/w/cpp/utility/functional/invoke
namespace kspc {
  /// @addtogroup utility
  /// @{

  /// @cond
  namespace detail {
    template <class>
    inline constexpr bool is_reference_wrapper_v = false;

    template <class U>
    inline constexpr bool is_reference_wrapper_v<std::reference_wrapper<U>> = true;

    template <class C, class Pointed, class T1, class... Args>
    constexpr decltype(auto) invoke_memptr(Pointed C::*f, T1&& t1, Args&&... args) {
      if constexpr (std::is_function_v<Pointed>) {
        if constexpr (std::is_base_of_v<C, std::decay_t<T1>>)
          return (std::forward<T1>(t1).*f)(std::forward<Args>(args)...);
        else if constexpr (is_reference_wrapper_v<std::decay_t<T1>>)
          return (t1.get().*f)(std::forward<Args>(args)...);
        else
          return ((*std::forward<T1>(t1)).*f)(std::forward<Args>(args)...);
      } else {
        static_assert(std::is_object_v<Pointed> && sizeof...(args) == 0);
        if constexpr (std::is_base_of_v<C, std::decay_t<T1>>)
          return std::forward<T1>(t1).*f;
        else if constexpr (is_reference_wrapper_v<std::decay_t<T1>>)
          return t1.get().*f;
        else
          return (*std::forward<T1>(t1)).*f;
      }
    }
  } // namespace detail
  /// @endcond

  /// invoke
  template <class F, class... Args>
  constexpr std::invoke_result_t<F, Args...>
  invoke(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_v<F, Args...>) {
    if constexpr (std::is_member_pointer_v<std::decay_t<F>>)
      return detail::invoke_memptr(f, std::forward<Args>(args)...);
    else
      return std::forward<F>(f)(std::forward<Args>(args)...);
  }

  /// @}
} // namespace kspc

// clang-format off

// sum
namespace kspc {
  /// @addtogroup numeric
  /// @{

  /// @cond
  namespace detail {
    template <class P, class I>
    using projected_t = std::invoke_result_t<P&, iter_reference_t<I>>;

    template <class I, class T, class Op = std::plus<>, class P = identity_fn>
    struct sum_constraints : std::conjunction<
      std::is_invocable<P&, iter_reference_t<I>>,
      std::is_invocable<Op&, T, projected_t<P, I>>> {};

    template <class I, class S, class T, class Op = std::plus<>, class P = identity_fn,
              std::enable_if_t<std::conjunction_v<
                is_sentinel_for<S, I>,
                is_input_iterator<I>,
                sum_constraints<I, T, Op, P>>, std::nullptr_t> = nullptr>
    constexpr auto sum(I first, S last, T init, Op op = {}, P proj = {}) {
      using U = remove_cvref_t<std::invoke_result_t<Op&, T, projected_t<P, I>>>;
      U ret = std::move(init);
      for (; first != last; ++first) {
        ret = kspc::invoke(op, std::move(ret), kspc::invoke(proj, *first));
      }
      return ret;
    }
  } // namespace detail
  /// @endcond

  /// @brief `std::accumulate` without the initial value
  /// @note The order of the arguments `P`, `Op` is different from range-v3.
  template <class R, class P = identity_fn, class Op = std::plus<>,
            std::enable_if_t<is_input_range_v<R>, std::nullptr_t> = nullptr>
  constexpr auto sum(R&& r, P proj = {}, Op op = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r));

    auto first = begin(r);
    auto init = kspc::invoke(proj, *first++);
    return detail::sum(first, end(r), init, std::move(op), std::move(proj));
  }

  /// @}
} // namespace kspc

// innerp
namespace kspc {
  /// @addtogroup numeric
  /// @{

  /// @cond
  namespace detail {
    template <class I1, class I2, class T,
              class Op1 = std::plus<>,
              class Op2 = std::multiplies<>,
              class P1 = conj_fn,
              class P2 = identity_fn>
    struct innerp_constraints : std::conjunction<
      std::is_invocable<P1&, iter_value_t<I1>>,
      std::is_invocable<P2&, iter_value_t<I2>>,
      std::is_invocable<Op2&, projected_t<P1, I1>, projected_t<P2, I2>>,
      std::is_invocable<Op1&, T, std::invoke_result_t<Op2&, projected_t<P1, I1>, projected_t<P2, I2>>>> {};

    template <class I1, class S1, class I2, class S2, class T,
              class Op1 = std::plus<>,
              class Op2 = std::multiplies<>,
              class P1 = conj_fn,
              class P2 = identity_fn,
              std::enable_if_t<std::conjunction_v<
                is_sentinel_for<S1, I1>,
                is_sentinel_for<S2, I2>,
                is_input_iterator<I1>,
                is_input_iterator<I2>,
                innerp_constraints<I1, I2, T, Op1, Op2, P1, P2>>, std::nullptr_t> = nullptr>
    constexpr auto innerp(I1 first1, S1 last1, I2 first2, S2 last2, T init,
                          Op1 op1 = {},
                          Op2 op2 = {},
                          P1 proj1 = {},
                          P2 proj2 = {}) {
      using U = remove_cvref_t<std::invoke_result_t<
                  Op1&,
                  T,
                  std::invoke_result_t<
                    Op2&,
                    projected_t<P1, I1>,
                    projected_t<P2, I2>>>>;
      U ret = std::move(init);
      for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
        ret = kspc::invoke(
                op1,
                std::move(ret),
                kspc::invoke(
                  op2,
                  kspc::invoke(proj1, *first1),
                  kspc::invoke(proj2, *first2)));
      }
      return ret;
    }
  } // namespace detail
  /// @endcond

  /// @brief `std::inner_product` without the initial value
  /// @note The order of the arguments `P#`, `Op#` is different from range-v3.
  template <class R1, class R2,
            class P1 = conj_fn,
            class P2 = identity_fn,
            class Op1 = std::plus<>,
            class Op2 = std::multiplies<>,
            std::enable_if_t<std::conjunction_v<
              is_input_range<R1>,
              is_input_range<R2>>, std::nullptr_t> = nullptr>
  constexpr auto innerp(R1&& r1, R2&& r2,
                        P1 proj1 = {},
                        P2 proj2 = {},
                        Op1 op1 = {},
                        Op2 op2 = {}) {
    using std::begin, std::end, std::empty; // for ADL
    assert(!empty(r1));
    assert(!empty(r2));

    auto first1 = begin(r1);
    auto first2 = begin(r2);
    auto init = kspc::invoke(
                  op2,
                  kspc::invoke(proj1, *first1++),
                  kspc::invoke(proj2, *first2++));
    return detail::innerp(
             first1, end(r1), first2, end(r2), init,
             std::move(op1),
             std::move(op2),
             std::move(proj1),
             std::move(proj2));
  }

  /// @brief norm
  template <class R,
            class P1 = conj_fn,
            class P2 = identity_fn,
            class Op1 = std::plus<>,
            class Op2 = std::multiplies<>,
            std::enable_if_t<is_input_range_v<R>, std::nullptr_t> = nullptr>
  constexpr auto norm(R&& r,
                      P1 proj1 = {},
                      P2 proj2 = {},
                      Op1 op1 = {},
                      Op2 op2 = {}) {
    return std::sqrt(innerp(r, r, std::move(proj1), std::move(proj2), std::move(op1), std::move(op2)));
  }

  /// @}
} // namespace kspc

// clang-format on

// arithmetic operators for `std::array` and `std::vector`
namespace kspc {
  /// @addtogroup numeric
  /// @{

  inline namespace arithmetic_ops {
    // std::array

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto& operator+(const std::array<T, N>& x) {
      return x;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(const std::array<T, N>& x) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = -x[j];
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(std::array<T, N>&& x) {
      for (std::size_t j = 0; j < N; ++j) x[j] = -x[j];
      return std::move(x);
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator+(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = x[j] + y[j];
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(const std::array<T, N>& x, const std::array<T, N>& y) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = x[j] - y[j];
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const T& val, const std::array<T, N>& x) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = val * x[j];
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const T& val, std::array<T, N>&& x) {
      for (std::size_t j = 0; j < N; ++j) x[j] = val * x[j];
      return std::move(x);
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const std::array<T, N>& x, const T& val) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = x[j] * val;
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(std::array<T, N>&& x, const T& val) {
      for (std::size_t j = 0; j < N; ++j) x[j] = x[j] * val;
      return std::move(x);
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator/(const std::array<T, N>& x, const T& val) {
      std::array<T, N> ret{};
      for (std::size_t j = 0; j < N; ++j) ret[j] = x[j] / val;
      return ret;
    }

    template <class T, std::size_t N,
              std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator/(std::array<T, N>&& x, const T& val) {
      for (std::size_t j = 0; j < N; ++j) x[j] = x[j] / val;
      return std::move(x);
    }

    // std::vector

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto& operator+(const std::vector<T>& x) {
      return x;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(const std::vector<T>& x) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = -x[j];
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(std::vector<T>&& x) {
      const auto n = std::size(x);
      for (std::size_t j = 0; j < n; ++j) x[j] = -x[j];
      return std::move(x);
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator+(const std::vector<T>& x, const std::vector<T>& y) {
      const auto n = std::min(std::size(x), std::size(y));
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = x[j] + y[j];
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator-(const std::vector<T>& x, const std::vector<T>& y) {
      const auto n = std::min(std::size(x), std::size(y));
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = x[j] - y[j];
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const T& val, const std::vector<T>& x) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = val * x[j];
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const T& val, std::vector<T>&& x) {
      const auto n = std::size(x);
      for (std::size_t j = 0; j < n; ++j) x[j] = val * x[j];
      return std::move(x);
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(const std::vector<T>& x, const T& val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = x[j] * val;
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator*(std::vector<T>&& x, const T& val) {
      const auto n = std::size(x);
      for (std::size_t j = 0; j < n; ++j) x[j] = x[j] * val;
      return std::move(x);
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator/(const std::vector<T>& x, const T& val) {
      const auto n = std::size(x);
      std::vector<T> ret(n);
      for (std::size_t j = 0; j < n; ++j) ret[j] = x[j] / val;
      return ret;
    }

    template <class T, std::enable_if_t<std::is_arithmetic_v<T>, std::nullptr_t> = nullptr>
    constexpr auto operator/(std::vector<T>&& x, const T& val) {
      const auto n = std::size(x);
      for (std::size_t j = 0; j < n; ++j) x[j] = x[j] / val;
      return std::move(x);
    }
  } // namespace arithmetic_ops

  /// @}
} // namespace kspc
