/// @file math.hpp
#pragma once
#include <algorithm> // all_of
#include <array>
#include <cmath> // exp, pow, cosh, sinh, etc.
#include <complex>
#include <functional> // invoke
#include <vector>
#include <kspc/math_basics.hpp>

namespace kspc {
  /// @addtogroup approx
  /// @{

  // approximate comparison

  /// @cond
  namespace detail {
    template <typename T,
              std::enable_if_t<std::is_convertible_v<T, double>, std::nullptr_t> = nullptr>
    constexpr double calclate_margin( //
      const T& value, const double epsrel, const double epsabs) {
      using std::isinf, std::abs, std::max; // for ADL
      return max(max(epsabs, 0.0), abs(isinf(value) ? 0.0 : value) * max(epsrel, 0.0));
    }
  } // namespace detail
  /// @endcond

  /// default epsilon
  template <typename T>
  inline constexpr T eps_v = std::enable_if_t<std::is_arithmetic_v<T>, T>(1e-6);

  /// alias for `eps_v<double>`
  inline constexpr double eps = eps_v<double>;

  /// %approx
  template <typename T, typename = void>
  struct approx {
    static_assert(always_false<T>);
  };

  /// partial specialization of @link approx approx @endlink
  template <typename T>
  struct approx<T, std::enable_if_t<std::is_convertible_v<T, double>>> {
  private:
    const T value_;
    const double margin_ = eps;

  public:
    constexpr explicit approx(const T& value, const double epsrel = eps, const double epsabs = 0.0)
      : value_(value), margin_(detail::calclate_margin(value_, epsrel, epsabs)) {}
    constexpr explicit approx(T&& value, const double epsrel = eps, const double epsabs = 0.0)
      : value_(std::move(value)), margin_(detail::calclate_margin(value_, epsrel, epsabs)) {}

    template <typename U,
              std::enable_if_t<std::is_convertible_v<U, double>, std::nullptr_t> = nullptr>
    friend inline constexpr bool
    operator<(const U& x, const approx& y) noexcept(noexcept(x + y.margin_ < y.value_)) {
      // avoid the subtraction with infinity
      return x + y.margin_ < y.value_;
    }
    template <typename U,
              std::enable_if_t<std::is_convertible_v<U, double>, std::nullptr_t> = nullptr>
    friend inline constexpr bool
    operator>(const U& x, const approx& y) noexcept(noexcept(x > y.value_ + y.margin_)) {
      // avoid the subtraction with infinity
      return x > y.value_ + y.margin_;
    }
    // boilerplate
    template <typename U>
    friend inline constexpr bool //
    operator<=(const U& x, const approx& y) noexcept(noexcept(!(x > y))) {
      return !(x > y);
    }
    template <typename U>
    friend inline constexpr bool //
    operator>=(const U& x, const approx& y) noexcept(noexcept(!(x < y))) {
      return !(x < y);
    }
    template <typename U>
    friend inline constexpr bool
    operator!=(const U& x, const approx& y) noexcept(noexcept((x < y) || (x > y))) {
      return x < y || x > y;
    }
    template <typename U>
    friend inline constexpr bool //
    operator==(const U& x, const approx& y) noexcept(noexcept(!(x != y))) {
      return !(x != y);
    }
  }; // struct approx

  /// deduction guide for @link approx approx @endlink
  template <typename T>
  approx(T, const double = eps, const double = 0.0) -> approx<T>;

  /// partial specialization of @link approx approx @endlink
  // TODO: comparison between floating-point and complex
  template <typename T>
  struct approx<std::complex<T>, std::enable_if_t<std::is_convertible_v<T, double>>> {
  private:
    const std::complex<T> value_;
    const double margin_ = eps;

  public:
    constexpr explicit approx(const std::complex<T>& value, const double epsrel = eps,
                              const double epsabs = 0.0)
      : value_(value), margin_(detail::calclate_margin(std::abs(value_), epsrel, epsabs)) {}
    constexpr explicit approx(std::complex<T>&& value, const double epsrel = eps,
                              const double epsabs = 0.0)
      : value_(std::move(value)),
        margin_(detail::calclate_margin(std::abs(value_), epsrel, epsabs)) {}

    template <typename U,
              std::enable_if_t<std::is_convertible_v<U, double>, std::nullptr_t> = nullptr>
    friend inline constexpr bool //
    operator==(const std::complex<U>& x, const approx& y) {
      using std::abs; // for ADL
      return std::abs(y.value_ - x) <= y.margin_;
    }
    // boilerplate
    template <typename U>
    friend inline constexpr bool //
    operator!=(const std::complex<U>& x, const approx& y) {
      return !(x == y);
    }
  }; // struct approx

  /// deduction guide for @link approx approx @endlink
  template <typename T>
  approx(std::complex<T>, const double = eps, const double = 0.0) -> approx<std::complex<T>>;

  // function objects for approximate comparison

#define KSPC_DEFINE_APPROXIMATE_COMPARISON(Name, Op)                                               \
  struct Name {                                                                                    \
  private:                                                                                         \
    const double epsrel_ = eps;                                                                    \
    const double epsabs_ = 0.0;                                                                    \
                                                                                                   \
  public:                                                                                          \
    constexpr Name() {}                                                                            \
    constexpr explicit Name(const double epsrel, const double epsabs = 0.0)                        \
      : epsrel_(epsrel), epsabs_(epsabs) {}                                                        \
                                                                                                   \
    template <typename T, typename U>                                                              \
    constexpr bool operator()(const T& x, const U& y) const                                        \
      noexcept(noexcept(x Op approx(y, epsrel_, epsabs_))) {                                       \
      return x Op approx(y, epsrel_, epsabs_);                                                     \
    }                                                                                              \
  };

  /// %approx_eq
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_eq, ==)
  /// %approx_ne
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_ne, !=)
  /// %approx_lt
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_lt, <)
  /// %approx_gt
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_gt, >)
  /// %approx_le
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_le, <=)
  /// %approx_ge
  KSPC_DEFINE_APPROXIMATE_COMPARISON(approx_ge, >=)

#undef KSPC_DEFINE_APPROXIMATE_COMPARISON

  /// @}

  /// @addtogroup matrix
  /// @{

  // matrix algorithms

  /// @brief Check whether matrix is hermitian.
  /// @example hermitian.cpp
  template <typename M, typename Proj1 = conj_fn, typename Proj2 = identity,
            typename Comp = approx_eq,
            std::enable_if_t<is_sized_range_v<M>, std::nullptr_t> = nullptr>
  constexpr bool hermitian(const M& op, Proj1 proj1 = {}, Proj2 proj2 = {}, Comp comp = {}) {
    const std::size_t N = kspc::dim(op);
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = j; k < N; ++k) {
        // clang-format off
        if (!std::invoke(comp,
                         std::invoke(proj1, op[j * N + k]),
                         std::invoke(proj2, op[k * N + j])))
          return false;
        // clang-format on
      }
    }
    return true;
  }

  // matrix class

  namespace detail2 {
    /// %matrix_base
    // clang-format off
    template <typename Derived,
              std::enable_if_t<std::conjunction_v<
                std::is_class<Derived>,
                std::is_same<Derived, std::remove_cv_t<Derived>>>, std::nullptr_t> = nullptr>
    // clang-format on
    struct matrix_base {
    private:
      constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
      }

      constexpr const Derived& derived() const noexcept {
        return static_cast<const Derived&>(*this);
      }

      template <typename D>
      using size_type_impl = std::make_unsigned_t<range_difference_t<D>>;

    public:
      constexpr auto cbegin() const noexcept(noexcept(begin(derived()))) {
        return begin(derived());
      }

      constexpr auto cend() const noexcept(noexcept(end(derived()))) {
        return end(derived());
      }

      constexpr auto rbegin() //
        noexcept(noexcept(std::make_reverse_iterator(end(derived())))) {
        return std::make_reverse_iterator(end(derived()));
      }

      constexpr auto rbegin() const noexcept(noexcept(std::make_reverse_iterator(end(derived())))) {
        return std::make_reverse_iterator(end(derived()));
      }

      constexpr auto rend() //
        noexcept(noexcept(std::make_reverse_iterator(begin(derived())))) {
        return std::make_reverse_iterator(begin(derived()));
      }

      constexpr auto rend() const noexcept(noexcept(std::make_reverse_iterator(begin(derived())))) {
        return std::make_reverse_iterator(begin(derived()));
      }

      constexpr auto crbegin() const noexcept(noexcept(rbegin())) {
        return rbegin();
      }

      constexpr auto crend() const noexcept(noexcept(rend())) {
        return rend();
      }

      [[nodiscard]] constexpr bool empty() //
        noexcept(noexcept(begin(derived()) == end(derived()))) {
        return begin(derived()) == end(derived());
      }

      [[nodiscard]] constexpr bool empty() const
        noexcept(noexcept(begin(derived()) == end(derived()))) {
        return begin(derived()) == end(derived());
      }

      constexpr auto size() noexcept(noexcept(make_unsigned_v(end(derived()) - begin(derived())))) {
        return make_unsigned_v(end(derived()) - begin(derived()));
      }

      constexpr auto size() const
        noexcept(noexcept(make_unsigned_v(end(derived()) - begin(derived())))) {
        return make_unsigned_v(end(derived()) - begin(derived()));
      }

      constexpr decltype(auto) front() noexcept(noexcept(*begin(derived()))) {
        assert(!empty());
        return *begin(derived());
      }

      constexpr decltype(auto) front() const noexcept(noexcept(*begin(derived()))) {
        assert(!empty());
        return *begin(derived());
      }

      constexpr decltype(auto) back() noexcept(noexcept(*(std::prev(end(derived()))))) {
        assert(!empty());
        return *std::prev(end(derived()));
      }

      constexpr decltype(auto) back() const noexcept(noexcept(*(std::prev(end(derived()))))) {
        assert(!empty());
        return *std::prev(end(derived()));
      }

      template <typename D = Derived>
      constexpr decltype(auto) operator[](const size_type_impl<D> j) //
        noexcept(noexcept(begin(derived())[j])) {
        return begin(derived())[j];
      }

      template <typename D = Derived>
      constexpr decltype(auto) operator[](const size_type_impl<D> j) const
        noexcept(noexcept(begin(derived())[j])) {
        return begin(derived())[j];
      }

      template <typename D = Derived>
      constexpr decltype(auto) at(const size_type_impl<D> j) {
        if (j >= size()) throw std::out_of_range("out_of_range thrown at matrix_base::at");
        return (*this)[j];
      }

      template <typename D = Derived>
      constexpr decltype(auto) at(const size_type_impl<D> j) const {
        if (j >= size()) throw std::out_of_range("out_of_range thrown at matrix_base::at");
        return (*this)[j];
      }

      constexpr auto data() noexcept(noexcept(std::addressof(*begin(derived())))) {
        return std::addressof(*begin(derived()));
      }

      constexpr auto data() const noexcept(noexcept(std::addressof(*begin(derived())))) {
        return std::addressof(*begin(derived()));
      }
    }; // struct matrix_base
  }    // namespace detail2

  /// fixed-size matrix
  template <typename T, std::size_t N>
  struct matrix : detail2::matrix_base<matrix<T, N>> {
  private:
    std::array<T, N * N> instance_{};

  public:
    using iterator = iterator_t<std::array<T, N * N>>;
    using const_iterator = iterator_t<const std::array<T, N * N>>;
    using difference_type = iter_difference_t<iterator>;
    using size_type = std::make_unsigned_t<difference_type>;
    using reference = iter_reference_t<iterator>;
    using const_reference = iter_reference_t<const_iterator>;
    using value_type = iter_value_t<iterator>;

    constexpr matrix() {}

    template <typename... U,
              std::enable_if_t<1UL + sizeof...(U) == N * N, std::nullptr_t> = nullptr>
    constexpr explicit matrix(const T& init, U&&... args)
      : instance_{init, std::forward<decltype(args)>(args)...} {}

    template <typename... U,
              std::enable_if_t<1UL + sizeof...(U) == N * N, std::nullptr_t> = nullptr>
    constexpr explicit matrix(T&& init, U&&... args)
      : instance_{std::move(init), std::forward<decltype(args)>(args)...} {}

    constexpr auto begin() noexcept(noexcept(instance_.begin())) {
      return instance_.begin();
    }

    constexpr auto begin() const noexcept(noexcept(instance_.begin())) {
      return instance_.begin();
    }

    constexpr auto end() noexcept(noexcept(instance_.end())) {
      return instance_.end();
    }

    constexpr auto end() const noexcept(noexcept(instance_.end())) {
      return instance_.end();
    }

    constexpr void swap(matrix& other) noexcept(noexcept(instance_.swap(other.instance_))) {
      instance_.swap(other.instance_);
    }

    // member function:
    // [x] dim
    // [x] operator()(const size_type j, const size_type k)
    // [x] at(const size_type j, const size_type k)

    constexpr size_type dim() const noexcept {
      return N;
    }

    constexpr decltype(auto) operator()(const size_type j, const size_type k) //
      noexcept(noexcept((*this)[j * dim() + k])) {
      return (*this)[j * dim() + k];
    }

    constexpr decltype(auto) operator()(const size_type j, const size_type k) const
      noexcept(noexcept((*this)[j * dim() + k])) {
      return (*this)[j * dim() + k];
    }

    constexpr decltype(auto) at(const size_type j, const size_type k) {
      return at(j * dim() + k);
    }

    constexpr decltype(auto) at(const size_type j, const size_type k) const {
      return at(j * dim() + k);
    }
  }; // struct matrix

  /// deduction guide for @link matrix matrix @endlink
  template <typename T, typename... U,
            std::enable_if_t<std::conjunction_v<std::is_same<U, T>...>, std::nullptr_t> = nullptr>
  matrix(T, U...) -> matrix<T, isqrt(1UL + sizeof...(U))>;

  /// partial specialization of `fixed_size_array_size`
  template <typename T, std::size_t N>
  struct fixed_size_array_size<matrix<T, N>> : std::integral_constant<std::size_t, N * N> {};

  /// dynamic-size matrix
  template <typename T>
  struct ndmatrix : detail2::matrix_base<ndmatrix<T>> {
    using iterator = iterator_t<std::vector<T>>;
    using const_iterator = iterator_t<const std::vector<T>>;
    using difference_type = iter_difference_t<iterator>;
    using size_type = std::make_unsigned_t<difference_type>;
    using reference = iter_reference_t<iterator>;
    using const_reference = iter_reference_t<const_iterator>;
    using value_type = iter_value_t<iterator>;

  private:
    size_type dim_ = 0;
    std::vector<T> instance_{};

  public:
    constexpr ndmatrix() {}

    constexpr explicit ndmatrix(size_type dim) : dim_(dim), instance_(dim * dim) {}

    constexpr explicit ndmatrix(size_type dim, const T& init)
      : dim_(dim), instance_(dim * dim, init) {}

    // clang-format off
    template <typename I, typename S,
              std::enable_if_t<
                std::conjunction_v<is_sentinel_for<S, I>, is_input_iterator<I>>, std::nullptr_t> = nullptr>
    // clang-format on
    constexpr ndmatrix(I first, S last)
      : dim_(isqrt(static_cast<size_type>(std::distance(first, last)))), instance_(dim_ * dim_) {
      std::copy_n(first, instance_.size(), instance_.begin());
    }

    constexpr ndmatrix(std::initializer_list<T> l) : ndmatrix(std::begin(l), std::end(l)) {}

    constexpr auto begin() noexcept(noexcept(instance_.begin())) {
      return instance_.begin();
    }

    constexpr auto begin() const noexcept(noexcept(instance_.begin())) {
      return instance_.begin();
    }

    constexpr auto end() noexcept(noexcept(instance_.end())) {
      return instance_.end();
    }

    constexpr auto end() const noexcept(noexcept(instance_.end())) {
      return instance_.end();
    }

    constexpr void swap(ndmatrix& other) //
      noexcept(std::is_nothrow_swappable_v<size_type>&& noexcept(instance_.swap(other.instance_))) {
      using std::swap; // for ADL
      swap(dim_, other.dim_);
      instance_.swap(other.instance_);
    }

    // member function:
    // [x] dim
    // [x] operator()(const size_type j, const size_type k)
    // [x] at(const size_type j, const size_type k)
    // [x] reshape

    constexpr size_type dim() const noexcept {
      return dim_;
    }

    constexpr decltype(auto) operator()(const size_type j, const size_type k) //
      noexcept(noexcept((*this)[j * dim() + k])) {
      return (*this)[j * dim() + k];
    }

    constexpr decltype(auto) operator()(const size_type j, const size_type k) const
      noexcept(noexcept((*this)[j * dim() + k])) {
      return (*this)[j * dim() + k];
    }

    constexpr decltype(auto) at(const size_type j, const size_type k) {
      return at(j * dim() + k);
    }

    constexpr decltype(auto) at(const size_type j, const size_type k) const {
      return at(j * dim() + k);
    }

    constexpr void reshape(const size_type dim) noexcept(noexcept(instance_.resize())) {
      dim_ = dim;
      instance_.resize(dim * dim);
    }
  }; // struct ndmatrix

  /// deduction guide for @link ndmatrix ndmatrix @endlink
  template <typename I, typename S>
  ndmatrix(I, S) -> ndmatrix<iter_value_t<I>>;

  // matrix element

  /// @cond
  namespace detail {
    // clang-format off
    template <typename M, typename Vs>
    struct fixed_mel_constraints : std::conjunction<
        is_fixed_size_array<M>,
        is_fixed_size_array<Vs>,
        is_fixed_size_array<range_reference_t<Vs>>> {};
    // clang-format on
  } // namespace detail
    /// @endcond

  /// mel
  // clang-format off
  template <typename L, typename Vs,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<L>,
              is_range<Vs>, is_range<range_reference_t<Vs>>,
              detail::fixed_mel_constraints<L, Vs>>, std::nullptr_t> = nullptr>
  // clang-format on
  constexpr auto mel(const L& op, const Vs& vs) {
    constexpr std::size_t N = fixed_size_array_size_v<Vs>;
    static_assert(fixed_size_array_size_v<L> == N * N
                    && fixed_size_array_size_v<remove_cvref_t<range_reference_t<Vs>>> == N,
                  "the matrix size and the vector sizes are inconsistent");

    using T = std::common_type_t<range_reference_t<L>, range_reference_t<range_reference_t<Vs>>>;
    matrix<T, N> ret{};
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        ret(i, j) = innerp(vs[i], op, vs[j]);
      }
    }
    return ret;
  }

  /// @overload
  // clang-format off
  template <typename M, typename Vs,
            std::enable_if_t<std::conjunction_v<
              is_sized_range<M>,
              is_range<Vs>, is_range<range_reference_t<Vs>>,
              std::negation<detail::fixed_mel_constraints<M, Vs>>>, std::nullptr_t> = nullptr>
  // clang-format on
  constexpr auto mel(const M& op, const Vs& vs) {
    using std::size; // for ADL
    const std::size_t N = kspc::dim(op);
    assert(size(vs) == N);

    using T = std::common_type_t<range_reference_t<M>, range_reference_t<range_reference_t<Vs>>>;
    ndmatrix<T> ret(N);
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        ret(i, j) = innerp(vs[i], op, vs[j]);
      }
    }
    return ret;
  }

  /// @}

  /// @addtogroup math_constants
  /// @{

  /// @cond
  namespace detail {
    template <typename T>
    using enable_if_floating = std::enable_if_t<std::is_floating_point_v<T>, T>;
  } // namespace detail
    /// @endcond

  // clang-format off
  /// Imaginary unit
  template <typename T> inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, std::complex<T>> i_v{0, 1};
  /// The Euler's number
  template <typename T> inline constexpr T e_v          = detail::enable_if_floating<T>(2.718281828459045235360287471352662498L);
  /// log_2 e
  template <typename T> inline constexpr T log2e_v      = detail::enable_if_floating<T>(1.442695040888963407359924681001892137L);
  /// log_10 e
  template <typename T> inline constexpr T log10e_v     = detail::enable_if_floating<T>(0.434294481903251827651128918916605082L);
  /// log_e 2
  template <typename T> inline constexpr T ln2_v        = detail::enable_if_floating<T>(0.693147180559945309417232121458176568L);
  /// log_e 10
  template <typename T> inline constexpr T ln10_v       = detail::enable_if_floating<T>(2.302585092994045684017991454684364208L);
  /// pi
  template <typename T> inline constexpr T pi_v         = detail::enable_if_floating<T>(3.141592653589793238462643383279502884L);
  /// tau
  template <typename T> inline constexpr T tau_v        = detail::enable_if_floating<T>(6.283185307179586476925286766559005768L);
  /// sqrt 2
  template <typename T> inline constexpr T sqrt2_v      = detail::enable_if_floating<T>(1.414213562373095048801688724209698079L);
  /// sqrt 3
  template <typename T> inline constexpr T sqrt3_v      = detail::enable_if_floating<T>(1.732050807568877293527446341505872367L);
  /// sqrt 5
  template <typename T> inline constexpr T sqrt5_v      = detail::enable_if_floating<T>(2.236067977499789696409173668731276235L);
  /// The Euler-Mascheroni constant
  template <typename T> inline constexpr T egamma_v     = detail::enable_if_floating<T>(0.577215664901532860606512090082402431L);
  /// Exponential of the Euler-Mascheroni constant
  template <typename T> inline constexpr T exp_egamma_v = detail::enable_if_floating<T>(1.781072417990197985236504103107179549L);

  /// alias for `i_v<double>`
  inline constexpr std::complex<double> i = i_v<double>;
  /// alias for `e_v<double>`
  inline constexpr double e          = e_v<double>;
  /// alias for `log2e_v<double>`
  inline constexpr double log2e      = log2e_v<double>;
  /// alias for `log10e_v<double>`
  inline constexpr double log10e     = log10e_v<double>;
  /// alias for `ln2_v<double>`
  inline constexpr double ln2        = ln2_v<double>;
  /// alias for `ln10_v<double>`
  inline constexpr double ln10       = ln10_v<double>;
  /// alias for `pi_v<double>`
  inline constexpr double pi         = pi_v<double>;
  /// alias for `tau_v<double>`
  inline constexpr double tau        = tau_v<double>;
  /// alias for `sqrt2_v<double>`
  inline constexpr double sqrt2      = sqrt2_v<double>;
  /// alias for `sqrt3_v<double>`
  inline constexpr double sqrt3      = sqrt3_v<double>;
  /// alias for `sqrt5_v<double>`
  inline constexpr double sqrt5      = sqrt5_v<double>;
  /// alias for `egamma_v<double>`
  inline constexpr double egamma     = egamma_v<double>;
  /// alias for `exp_egamma_v<double>`
  inline constexpr double exp_egamma = exp_egamma_v<double>;
  // clang-format on

  /// @}

  /// @addtogroup math_functions
  /// @{

  /// The Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> f_FD(const T E, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (E - mu)) + 1.0);
  }

  /// Derivative of the Fermi distribution
  template <typename T>
  inline detail::enable_if_floating<T> Df_FD(const T E, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::cosh(0.5 * beta * (E - mu)), -2);
  }

  /// The Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> f_BE(const T E, const T beta, const T mu) {
    return 1.0 / (std::exp(beta * (E - mu)) - 1.0);
  }

  /// Derivative of the Bose distribution
  template <typename T>
  inline detail::enable_if_floating<T> Df_BE(const T E, const T beta, const T mu) {
    return -beta * std::pow(2.0 * std::sinh(0.5 * beta * (E - mu)), -2);
  }

  /// lerp (C++20)
  template <typename T, typename U>
  inline constexpr auto lerp(const T& a, const T& b, const U& t) noexcept(noexcept(a + t * (b - a)))
    -> decltype((a + t * (b - a))) {
    if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<U>) {
      if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) {
        return (1 - t) * a + t * b;
      }
    }
    return a + t * (b - a);
  }

  /// squared
  template <typename T>
  inline constexpr auto squared(const T& x) noexcept(noexcept(x* x)) -> decltype((x * x)) {
    return x * x;
  }

  /// cubed
  template <typename T>
  inline constexpr auto cubed(const T& x) noexcept(noexcept(x* x* x)) -> decltype((x * x * x)) {
    return x * x * x;
  }

  /// @}

  /// @addtogroup physics
  /// @{

  template <typename K, typename B,
            std::enable_if_t<is_range_v<K> && is_range_v<B>, std::nullptr_t> = nullptr>
  inline constexpr bool in_BZ(const K& k, const B& b) {
    return 2 * std::abs(innerp(k, b)) < innerp(b, b);
  }

  template <typename Bs, std::enable_if_t<is_range_v<Bs>, std::nullptr_t> = nullptr>
  inline constexpr auto make_in_BZ(const Bs& bs) {
    return [&bs](const auto& k) {
      using std::begin, std::end; // for ADL
      return std::all_of(begin(bs), end(bs), [&k](const auto& b) { return in_BZ(k, b); });
    };
  }

  /// @}
} // namespace kspc
