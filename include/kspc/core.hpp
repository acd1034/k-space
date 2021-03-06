/// @file core.hpp
#pragma once
#include <cassert>          // assert
#include <cstddef>          // size_t, ptrdiff_t, nullptr_t
#include <cstdint>          // int32_t
#include <initializer_list> // initializer_list
#include <tuple>            // tuple
#include <type_traits>      // enable_if_t, void_t, true_type, invoke_result, etc.
#include <utility>          // move, forward, pair, swap, exchange, declval

#include <complex>
#include <iosfwd>   // basic_ostream
#include <iterator> // begin, end, size, data
#include <limits>   // numeric_limits

/// @defgroup utility Utility
/// Utility inline variables, type aliases, type transformations and function objects

/// @defgroup iterator Iterator

/// @defgroup range Range

/// @defgroup io IO

/// @defgroup math Math

/// @defgroup matrix Matrix
/// @ingroup math

/// @defgroup numeric Numerical algorithms
/// @ingroup math

/// @defgroup approx Approximate comparison
/// @ingroup math

/// @defgroup math_constants Mathematical constants
/// @ingroup math

/// @defgroup math_functions Mathematical functions
/// @ingroup math

/// @defgroup linalg Linear algorithms
/// @ingroup math

/// @defgroup integration Quadrature routines
/// @ingroup math

/// @defgroup isoline Isoline
/// @ingroup math

/// @defgroup physics Physics

// Utility
namespace kspc {
  /// @addtogroup utility
  /// @{

  /// always_false
  template <typename...>
  inline constexpr bool always_false = false;

  /// always_true_type
  template <typename...>
  using always_true_type = std::true_type;

  /// %remove_cvref (C++20)
  template <typename T>
  struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  /// remove_cvref_t (C++20)
  template <typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;

  /// %type_identity (C++20)
  template <typename T>
  struct type_identity {
    using type = T;
  };

  /// type_identity_t (C++20)
  template <class T>
  using type_identity_t = typename type_identity<T>::type;

  /// %is_same_uncvref
  template <typename T, typename U>
  struct is_same_uncvref : std::is_same<remove_cvref_t<T>, remove_cvref_t<U>> {};

  /// helper variable template for `is_same_uncvref`
  template <typename T, typename U>
  inline constexpr bool is_same_uncvref_v = is_same_uncvref<T, U>::value;

  /// cast_if_needed
  template <typename U, typename T,
            std::enable_if_t<!is_same_uncvref_v<T, U>, std::nullptr_t> = nullptr>
  constexpr U cast_if_needed(T x) noexcept(noexcept(static_cast<U>(x))) {
    return static_cast<U>(x);
  }

  /// @overload
  template <typename U, typename T,
            std::enable_if_t<is_same_uncvref_v<T, U>, std::nullptr_t> = nullptr>
  constexpr T&& cast_if_needed(T&& x) noexcept {
    return std::forward<T>(x);
  }

  /// make_signed_v
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>, std::nullptr_t> = nullptr>
  constexpr std::make_signed_t<T>
  make_signed_v(T x) noexcept(noexcept(static_cast<std::make_signed_t<T>>(x))) {
    using U = std::make_signed_t<T>;
    assert(x <= static_cast<T>(std::numeric_limits<U>::max()));
    return static_cast<U>(x);
  }

  /// @overload
  template <typename T,
            std::enable_if_t<!std::is_unsigned_v<remove_cvref_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr T&& make_signed_v(T&& x) noexcept {
    return std::forward<T>(x);
  }

  /// make_unsigned_v
  template <typename T, std::enable_if_t<std::is_signed_v<T>, std::nullptr_t> = nullptr>
  constexpr std::make_unsigned_t<T>
  make_unsigned_v(T x) noexcept(noexcept(static_cast<std::make_unsigned_t<T>>(x))) {
    assert(x >= static_cast<T>(0));
    return static_cast<std::make_unsigned_t<T>>(x);
  }

  /// @overload
  template <typename T,
            std::enable_if_t<!std::is_signed_v<remove_cvref_t<T>>, std::nullptr_t> = nullptr>
  inline constexpr T&& make_unsigned_v(T&& x) noexcept {
    return std::forward<T>(x);
  }

  // detection idiom

  /// @cond
  namespace detail {
    template <typename AlwaysVoid, template <typename...> typename Op, typename... Args>
    struct is_detected_impl : std::false_type {};

    template <template <typename...> typename Op, typename... Args>
    struct is_detected_impl<std::void_t<Op<Args...>>, Op, Args...> : std::true_type {};
  } // namespace detail
  /// @endcond

  /// %is_detected
  template <template <typename...> typename Op, typename... Args>
  struct is_detected : detail::is_detected_impl<void, Op, Args...> {};

  /// helper variable template for `is_detected`
  template <template <typename...> typename Op, typename... Args>
  inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

  // type aliases to get the return type of operators

  /// @cond
  namespace detail {
    template <class T, class U>
    using minus_t = decltype(std::declval<T>() - std::declval<U>());

    template <typename T>
    using dereference_t = decltype(*std::declval<T>());

    template <typename T>
    using pre_increment_t = decltype(++std::declval<T>());

    template <typename T>
    using post_increment_t = decltype(std::declval<T>()++);

    template <typename T, typename U>
    using equal_to_t = decltype(std::declval<T>() == std::declval<U>());

    template <typename T, typename U>
    using not_equal_to_t = decltype(std::declval<T>() != std::declval<U>());
  } // namespace detail
    /// @endcond

  /// @}
} // namespace kspc

// clang-format off

// Iterator
namespace kspc {
  /// @addtogroup iterator
  /// @{

  // iterator associated types (first half):
  // [x] incrementable_traits
  // [x] indirectly_readable_traits
  // [x] iter_value_t
  // [x] iter_difference_t

  // incrementable_traits

  /// @cond
  namespace detail {
    template <class, class = void>
    struct incrementable_traits_impl {};

    template <class T>
    struct incrementable_traits_impl<
      T, std::enable_if_t<is_detected_v<minus_t, const T&, const T&> /* -> std::integral */>> {
      using difference_type = std::make_signed_t<minus_t<T, T>>;
    };

    // for incrementable_traits<T*>
    struct nil {};

    // for incrementable_traits<T*>
    template <class T>
    struct with_difference_type {
      using difference_type = T;
    };
  } // namespace detail
  /// @endcond

  /// %incrementable_traits
  template <class T, class = void>
  struct incrementable_traits : detail::incrementable_traits_impl<T> {};

  /// partial specialization of `incrementable_traits`
  template <class T>
  struct incrementable_traits<T*>
    : std::conditional_t<
        std::is_object_v<T>,
        detail::with_difference_type<std::ptrdiff_t>,
        detail::nil> {};

  /// partial specialization of `incrementable_traits`
  template <class T>
  struct incrementable_traits<T, std::void_t<typename T::difference_type>> {
    using difference_type = typename T::difference_type;
  };

  /// partial specialization of `incrementable_traits`
  template <class T>
  struct incrementable_traits<const T> : incrementable_traits<T> {};

  // indirectly_readable_traits

  /// @cond
  namespace detail {
    template <class, class = void>
    struct indirectly_readable_traits_impl {};

    template <class T>
    struct indirectly_readable_traits_impl<T, std::void_t<typename T::element_type>> {
      using value_type = std::remove_cv_t<typename T::element_type>;
    };
  } // namespace detail
  /// @endcond

  /// %indirectly_readable_traits
  template <class T, class = void>
  struct indirectly_readable_traits : detail::indirectly_readable_traits_impl<T> {};

  /// partial specialization of `indirectly_readable_traits`
  template <class T>
  struct indirectly_readable_traits<T*> {
    using value_type = std::remove_cv_t<T>;
  };

  /// partial specialization of `indirectly_readable_traits`
  template <class T>
  struct indirectly_readable_traits<T[]> {
    using value_type = std::remove_cv_t<T>;
  };

  /// partial specialization of `indirectly_readable_traits`
  template <class T, std::size_t N>
  struct indirectly_readable_traits<T[N]> {
    using value_type = std::remove_cv_t<T>;
  };

  /// partial specialization of `indirectly_readable_traits`
  template <class T>
  struct indirectly_readable_traits<T, std::void_t<typename T::value_type>> {
    using value_type = std::remove_cv_t<typename T::value_type>;
  };

  /// partial specialization of `indirectly_readable_traits`
  template <class T>
  struct indirectly_readable_traits<const T> : indirectly_readable_traits<T> {};

  /// @brief iter_difference_t
  /// @note Specialization of `std::iterator_traits` is NOT supported.
  template <typename I>
  using iter_difference_t = typename incrementable_traits<remove_cvref_t<I>>::difference_type;

  /// @brief iter_value_t
  /// @note Specialization of `std::iterator_traits` is NOT supported.
  template <typename I>
  using iter_value_t = typename indirectly_readable_traits<remove_cvref_t<I>>::value_type;

  // iterator concepts:
  // [x] is_dereferenceable (exposition only)
  // [x] is_weakly_incrementable
  // [x] is_input_or_output_iterator = is_dereferenceable && is_weakly_incrementable
  // [x] is_indirectly_readable
  // [x] is_input_iterator = is_input_or_output_iterator && is_indirectly_readable
  // [x] is_weakly_equality_comparable_with<T, U> (exposition only)
  // [x] is_sentinel_for<S, I> = is_input_or_output_iterator<I> && is_weakly_equality_comparable_with<S, I>

  /// @cond
  namespace detail {
    // is_dereferenceable

    template <typename T>
    struct is_dereferenceable : is_detected<dereference_t, T&> /* -> not-void (in particular) */ {};

    template <typename T>
    inline constexpr bool is_dereferenceable_v = is_dereferenceable<T>::value;

    // necessity for is_weakly_incrementable

    template <typename T>
    struct is_pre_incrementable : is_detected<pre_increment_t, T&> {};

    // WORKAROUND: ISO C++17 does not allow incrementing expression of type bool [-Wincrement-bool]
    template <>
    struct is_pre_incrementable<bool> : std::false_type {};

    template <typename T>
    inline constexpr bool is_pre_incrementable_v = is_pre_incrementable<T>::value;

    template <typename T>
    struct is_post_incrementable : is_detected<post_increment_t, T&> {};

    // WORKAROUND: ISO C++17 does not allow incrementing expression of type bool [-Wincrement-bool]
    template <>
    struct is_post_incrementable<bool> : std::false_type {};

    template <typename T>
    inline constexpr bool is_post_incrementable_v = is_post_incrementable<T>::value;
  } // namespace detail
  /// @endcond

  /// %is_weakly_incrementable
  template <typename I>
  struct is_weakly_incrementable
    : std::conjunction<
        // is_movable<I>,
        is_detected<iter_difference_t, I>, // -> is-signed-integer-like
        detail::is_pre_incrementable<I>,   // -> std::same_as<I&>, not required to be equality-preserving
        detail::is_post_incrementable<I>   // not required to be equality-preserving
        > {};

  /// helper variable template for `is_weakly_incrementable`
  template <typename I>
  inline constexpr bool is_weakly_incrementable_v = is_weakly_incrementable<I>::value;

  /// %is_input_or_output_iterator
  template <typename I>
  struct is_input_or_output_iterator
    : std::conjunction<
        detail::is_dereferenceable<I>,
        is_weakly_incrementable<I>
        > {};

  /// helper variable template for `is_input_or_output_iterator`
  template <typename I>
  inline constexpr bool is_input_or_output_iterator_v = is_input_or_output_iterator<I>::value;

  /// %is_indirectly_readable
  template <typename T>
  struct is_indirectly_readable
    : std::conjunction<
        is_detected<iter_value_t, const remove_cvref_t<T>>,
        detail::is_dereferenceable<const remove_cvref_t<T>>
        > {};

  /// helper variable template for `is_indirectly_readable`
  template <typename T>
  inline constexpr bool is_indirectly_readable_v = is_indirectly_readable<T>::value;

  /// %is_input_iterator
  template <typename I>
  struct is_input_iterator
    : std::conjunction<
        is_input_or_output_iterator<I>,
        is_indirectly_readable<I>
        > {};

  /// helper variable template for `is_input_iterator`
  template <typename I>
  inline constexpr bool is_input_iterator_v = is_input_iterator<I>::value;

  /// @cond
  namespace detail {
    // is_weakly_equality_comparable_with

    template <typename T, typename U,
              typename T2 = std::remove_reference_t<T>,
              typename U2 = std::remove_reference_t<U>>
    struct is_weakly_equality_comparable_with
      : std::conjunction<
          is_detected<equal_to_t, const T2&, const U2&>,     // -> boolean-testable
          is_detected<equal_to_t, const U2&, const T2&>,     // -> boolean-testable
          is_detected<not_equal_to_t, const T2&, const U2&>, // -> boolean-testable
          is_detected<not_equal_to_t, const U2&, const T2&>  // -> boolean-testable
          > {};

    template <typename T, typename U>
    inline constexpr bool is_weakly_equality_comparable_with_v = is_weakly_equality_comparable_with<T, U>::value;
  } // namespace detail
  /// @endcond

  /// %is_sentinel_for
  template <typename S, typename I>
  struct is_sentinel_for
    : std::conjunction<
        // is_semiregular<S>,
        is_input_or_output_iterator<I>,
        detail::is_weakly_equality_comparable_with<S, I>
        > {};

  /// helper variable template for `is_sentinel_for`
  template <typename S, typename I>
  inline constexpr bool is_sentinel_for_v = is_sentinel_for<S, I>::value;

  // iterator associated types (second half):
  // [x] iter_reference_t
  // [ ] iter_rvalue_reference_t (need `iter_move`)
  // [ ] iter_common_reference_t (need `common_reference_t`)

  /// iter_reference_t
  template <typename I, std::enable_if_t<detail::is_dereferenceable_v<I>, std::nullptr_t> = nullptr>
  using iter_reference_t = detail::dereference_t<I&>;

  /// @}
} // namespace kspc

// Range
namespace kspc {
  /// @addtogroup range
  /// @{

  /// @cond
  namespace detail_adl {
    using std::begin, std::end, std::size, std::data, std::swap; // for ADL

    template <typename C>
    constexpr auto adl_begin(C&& c) noexcept(noexcept(begin(std::forward<C>(c))))
      -> decltype(begin(std::forward<C>(c))) {
      return begin(std::forward<C>(c));
    }

    template <typename C>
    constexpr auto adl_end(C&& c) noexcept(noexcept(end(std::forward<C>(c))))
      -> decltype(end(std::forward<C>(c))) {
      return end(std::forward<C>(c));
    }

    template <typename C>
    constexpr auto adl_size(C&& c) noexcept(noexcept(size(std::forward<C>(c))))
      -> decltype(size(std::forward<C>(c))) {
      return size(std::forward<C>(c));
    }

    template <typename C>
    constexpr auto adl_data(C&& c) noexcept(noexcept(data(std::forward<C>(c))))
      -> decltype(data(std::forward<C>(c))) {
      return data(std::forward<C>(c));
    }

    template <typename T>
    constexpr void adl_swap(T&& lhs, T&& rhs) //
      noexcept(noexcept(swap(std::forward<T>(lhs), std::forward<T>(rhs)))) {
      swap(std::forward<T>(lhs), std::forward<T>(rhs));
    }
  } // namespace detail_adl
  /// @endcond

  using detail_adl::adl_begin, detail_adl::adl_end, detail_adl::adl_size, detail_adl::adl_data, detail_adl::adl_swap;

  // range concepts (first half):
  // [x] is_range
  // [x] is_sized_range

  /// @cond
  namespace detail {
    template <typename T>
    using adl_begin_t = decltype(adl_begin(std::declval<T>()));

    template <typename T>
    using adl_end_t = decltype(adl_end(std::declval<T>()));

    template <typename T>
    using adl_size_t = decltype(adl_size(std::declval<T>()));

    template <typename T>
    using adl_data_t = decltype(adl_data(std::declval<T>()));
  } // namespace detail
  /// @endcond

  /// %is_range
  template <typename R>
  struct is_range
    : std::conjunction<
        is_detected<detail::adl_begin_t, R&>,
        is_detected<detail::adl_end_t, R&>
        > {};

  /// helper variable template for `is_range`
  template <typename R>
  inline constexpr bool is_range_v = is_range<R>::value;

  /// %is_sized_range
  template <typename R>
  struct is_sized_range
    : std::conjunction<
        is_range<R>,
        is_detected<detail::adl_size_t, R&>
        > {};

  /// helper variable template for `is_sized_range`
  template <typename R>
  inline constexpr bool is_sized_range_v = is_sized_range<R>::value;

  // range associated types:
  // [x] iterator_t
  // [x] sentinel_t
  // [x] range_difference_t (alias)
  // [x] range_size_t
  // [x] range_value_t (alias)
  // [x] range_reference_t (alias)
  // [ ] range_rvalue_reference_t (need `iter_rvalue_reference_t`)

  /// iterator_t
  template <typename R>
  using iterator_t = detail::adl_begin_t<R&>;

  /// sentinel_t
  template <typename R, std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
  using sentinel_t = detail::adl_end_t<R&>;

  /// range_difference_t
  template <typename R, std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
  using range_difference_t = iter_difference_t<iterator_t<R>>;

  /// range_size_t
  template <typename R, std::enable_if_t<is_sized_range_v<R>, std::nullptr_t> = nullptr>
  using range_size_t = detail::adl_size_t<R&>;

  /// range_value_t
  template <typename R, std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
  using range_value_t = iter_value_t<iterator_t<R>>;

  /// range_reference_t
  template <typename R, std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
  using range_reference_t = iter_reference_t<iterator_t<R>>;

  // range concepts (second half):
  // [x] is_input_range<R> = is_range<R> && is_input_iterator<iterator_t<R>>

  /// %is_input_range
  template <typename R>
  struct is_input_range : std::conjunction<is_range<R>, is_input_iterator<iterator_t<R>>> {};

  /// helper variable template for `is_input_range`
  template <typename R>
  inline constexpr bool is_input_range_v = is_input_range<R>::value;

  /// @}
} // namespace kspc

// clang-format on

// IO
namespace kspc {
  /// @addtogroup io
  /// @{

  inline namespace io {
    /// Stream insertion for range
    template <typename CharT, typename Traits, typename R,
              std::enable_if_t<is_range_v<R>, std::nullptr_t> = nullptr>
    auto& operator<<(std::basic_ostream<CharT, Traits>& os, const R& r) {
      const char* dlm = "";
      for (const auto& x : r) os << std::exchange(dlm, " ") << x;
      return os;
    }
  } // namespace io

  /// @}
} // namespace kspc

// projection
namespace kspc {
  /// @addtogroup math
  /// @{

  // identity function

  /// %identity_fn
  struct identity_fn {
    using is_transparent = void;

    template <typename T>
    constexpr T&& operator()(T&& t) const noexcept {
      return std::forward<T>(t);
    }
  }; // struct identity_fn

  inline namespace cpo {
    /// identity
    inline constexpr identity_fn identity{};
  } // namespace cpo

  // consistent complex access

  /// %is_complex
  template <typename T>
  struct is_complex : std::false_type {};

  /// partial specialization of `is_complex`
  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

  /// helper variable template for `is_complex`
  template <typename T>
  inline constexpr bool is_complex_v = is_complex<T>::value;

  /// %conj_fn
  struct conj_fn {
    using is_transparent = void;

    template <typename T, std::enable_if_t<is_complex_v<T>, std::nullptr_t> = nullptr>
    constexpr T operator()(const T& x) const noexcept(noexcept(real(x), -imag(x))) {
      return {real(x), -imag(x)};
    }

    // same as `identity`
    template <typename T,
              std::enable_if_t<!is_complex_v<remove_cvref_t<T>>, std::nullptr_t> = nullptr>
    constexpr T&& operator()(T&& x) const noexcept {
      return std::forward<T>(x);
    }
  }; // struct conj_fn

  inline namespace cpo {
    /// conj
    inline constexpr conj_fn conj{};
  } // namespace cpo

  /// @}
} // namespace kspc
