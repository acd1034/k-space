/// @file approx.hpp
#pragma once
#include <ranges>
#include <kspc/core.hpp>

namespace kspc {
  /// @addtogroup physics
  /// @{

  /// @cond
  namespace detail {
    template <class T, class U>
    concept addable = requires(T t, const U& u) {
      { t + u } -> std::convertible_to<T>;
    };
    template <class T, class U>
    concept subtractable = requires(T t, const U& u) {
      { t - u } -> std::convertible_to<T>;
    };
  } // namespace detail
  /// @endcond

  /// `std::ranges::iota_view`-like view for classes with `operator+`
  template <std::copyable T, std::copyable U>
  requires detail::addable<T, U>
  struct kappa_view : std::ranges::view_interface<kappa_view<T, U>> {
  private:
    T init_ = T();
    U update_ = U();
    std::ptrdiff_t bound_ = 0;

    /// iterator for `kappa_view`
    struct iterator;

  public:
    // ctor
    kappa_view() requires std::default_initializable<T> and std::default_initializable<U>
    = default;
    constexpr kappa_view(T t, U u, std::ptrdiff_t n)
      : init_(std::move(t)), update_(std::move(u)), bound_(std::move(n)) {
      assert(bound_ >= 0);
    }

    // range
    constexpr iterator begin() const {
      return iterator(*this);
    }
    constexpr auto end() const {
      return std::default_sentinel;
    }
    constexpr auto size() const {
      return static_cast<std::size_t>(bound_);
    }
  };

  /// @cond
  namespace detail {
    template <class, class>
    struct kappa_iterator_category {};

    template <std::copyable T, std::copyable U>
    requires detail::addable<T, U>
    struct kappa_iterator_category<T, U> {
      using iterator_category =
        std::conditional_t<detail::subtractable<T, U>, std::bidirectional_iterator_tag,
                           std::forward_iterator_tag>;
    };
  } // namespace detail
  /// @endcond

  template <std::copyable T, std::copyable U>
  requires detail::addable<T, U>
  struct kappa_view<T, U>::iterator : detail::kappa_iterator_category<T, U> {
  private:
    const kappa_view* parent_ = nullptr;
    T current_ = T();
    std::ptrdiff_t count_ = 0;

  public:
    // using iterator_category = inherited;
    using iterator_concept =
      std::conditional_t<detail::subtractable<T, U>, std::bidirectional_iterator_tag,
                         std::forward_iterator_tag>;
    using difference_type = std::ptrdiff_t;
    using value_type = T;

    // ctor
    iterator() requires std::default_initializable<T>
    = default;
    constexpr iterator(const kappa_view& p)
      : parent_(std::addressof(p)), current_(p.init_), count_(p.bound_) {
      assert(count_ >= 0);
    }

    // observer
    constexpr std::ptrdiff_t count() const noexcept {
      return count_;
    }

    // iterator
    constexpr const T& operator*() const {
      assert(count_ >= 0);
      return current_;
    }

    constexpr iterator& operator++() {
      assert(count_ >= 0);
      current_ = std::move(current_) + parent_->update_;
      --count_;
      return *this;
    }
    constexpr iterator operator++(int) {
      assert(count_ >= 0);
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    constexpr iterator& operator--() requires detail::subtractable<T, U> {
      assert(count_ < parent_->bound_);
      current_ = std::move(current_) - parent_->update_;
      ++count_;
      return *this;
    }
    constexpr iterator operator--(int) requires detail::subtractable<T, U> {
      assert(count_ < parent_->bound_);
      auto tmp = *this;
      --*this;
      return tmp;
    }

    friend constexpr bool operator==(const iterator& x,
                                     const iterator& y) requires std::equality_comparable<T> {
      return x.current_ == y.current_ and x.count_ == y.count_;
    }
    friend constexpr bool operator==(const iterator& x, std::default_sentinel_t) {
      return x.count_ == 0;
    }
  };

  /// @}
} // namespace kspc
