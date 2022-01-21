/// @file linalg.hpp
#pragma once
#include <algorithm> // fill
#include <array>
#include <cmath>      // sqrt, round
#include <functional> // invoke
#include <vector>
#include <kspc/core.hpp> // is_range, identity_fu, conj_fn

// dim
namespace kspc {
  /// @addtogroup matrix
  /// @{

  // fixed-size array support

  /// %fixed_size_array_size
  template <typename>
  struct fixed_size_array_size {};

  /// partial specialization of `fixed_size_array_size`
  template <typename T, std::size_t N>
  struct fixed_size_array_size<T[N]> : std::integral_constant<std::size_t, N> {};

  /// partial specialization of `fixed_size_array_size`
  template <typename T, std::size_t N>
  struct fixed_size_array_size<std::array<T, N>> : std::integral_constant<std::size_t, N> {};

  /// helper variable template for `fixed_size_array_size`
  template <typename T>
  inline constexpr auto fixed_size_array_size_v = fixed_size_array_size<T>::value;

  /// %is_fixed_size_array
  template <typename T, typename = void>
  struct is_fixed_size_array : std::false_type {};

  /// partial specialization of `is_fixed_size_array`
  template <typename T>
  struct is_fixed_size_array<T, std::void_t<decltype(fixed_size_array_size<T>::value)>>
    : std::true_type {};

  /// @brief helper variable template for `is_fixed_size_array`
  /// @details To make this true, partially/fully specialize `fixed_size_array_size`.
  template <typename T>
  inline constexpr bool is_fixed_size_array_v = is_fixed_size_array<T>::value;

  // matrix dimension

  /// compile-time sqrt for unsigned integer
  inline constexpr std::size_t isqrt(const std::size_t n) noexcept {
    std::size_t l = 0, r = n;
    while (r - l > 1) {
      const std::size_t mid = l + (r - l) / 2;
      if (mid * mid <= n) {
        l = mid;
      } else {
        r = mid;
      }
    }
    return l;
  }

  /// fixed_size_matrix_dim_v
  template <typename T>
  inline constexpr auto fixed_size_matrix_dim_v = isqrt(fixed_size_array_size<T>::value);

  /// dim
  template <typename M, std::enable_if_t<is_sized_range_v<M>, std::nullptr_t> = nullptr>
  inline auto dim(const M& m) {
    return static_cast<decltype(adl_size(m))>(std::round(std::sqrt(adl_size(m))));
  }

  /// @}
} // namespace kspc

// mapping
namespace kspc {
  /// @addtogroup matrix
  /// @{

  /// %mapping_row_major
  struct mapping_row_major {
  private:
    std::size_t lda_{};

  public:
    using size_type = std::size_t;
    constexpr mapping_row_major() = default;
    constexpr explicit mapping_row_major(const size_type lda) : lda_(lda) {}

    constexpr size_type operator()(const size_type i, const size_type j) const noexcept {
      return lda_ * i + j;
    }
  }; // struct mapping_row_major

  /// %mapping_transpose
  template <typename Mapping>
  struct mapping_transpose {
  private:
    Mapping mapping_{};

  public:
    using size_type = std::size_t;
    constexpr mapping_transpose() = default;
    constexpr explicit mapping_transpose(const Mapping& mapping) : mapping_(mapping) {}
    constexpr explicit mapping_transpose(Mapping&& mapping) : mapping_(std::move(mapping)) {}

    constexpr size_type operator()(const size_type i, const size_type j) const noexcept {
      return mapping_(j, i);
    }
  }; // struct mapping_transpose

  /// deduction guide for @link mapping_transpose mapping_transpose @endlink
  template <typename Mapping>
  mapping_transpose(Mapping) -> mapping_transpose<Mapping>;

  /// mapping_column_major
  inline constexpr auto mapping_column_major(const std::size_t lda) {
    return mapping_transpose(mapping_row_major(lda));
  }

  /// @}
} // namespace kspc

// clang-format off

// matrix_copy
namespace kspc {
  /// @addtogroup linalg
  /// @{

  /// matrix_copy
  template <class InMat, class OutMat, class M1, class M2, class P1 = identity_fn>
  void matrix_copy(const InMat& A, OutMat& B, M1&& map1, M2&& map2, P1&& proj1 = {}) {
    using std::size; // for ADL
    const std::size_t n = kspc::dim(A);
    assert(size(A) == n * n);
    assert(size(B) == n * n);

    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        B[map2(j, k)] = std::invoke(proj1, A[map1(j, k)]);
      }
    }
  }

  /// @}
} // namespace kspc

// unitary_transform
namespace kspc {
  /// @addtogroup linalg
  /// @{

  /// matrix_product
  template <class InMat1, class InMat2, class OutMat, class M1, class M2, class M3, class P1 = identity_fn, class P2 = identity_fn>
  void matrix_product(const InMat1& A, const InMat2& B, OutMat& C, M1&& map1, M2&& map2, M3&& map3, P1&& proj1 = {}, P2&& proj2 = {}) {
    using std::size; // for ADL
    const std::size_t n = kspc::dim(A);
    assert(size(A) == n * n);
    assert(size(B) == n * n);
    assert(size(C) == n * n);

    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t l = 0; l < n; ++l) {
        for (std::size_t k = 0; k < n; ++k) {
          C[map3(j, k)] += std::invoke(proj1, A[map1(j, l)]) * std::invoke(proj2, B[map2(l, k)]);
        }
      }
    }
  }

  /// unitary_transform
  template <class InOutMat, class InMat, class Work, class M2, class M3, class P1 = conj_fn, class P2 = identity_fn, class P3 = identity_fn>
  std::enable_if_t<std::conjunction_v<is_range<InOutMat>, is_range<InMat>, is_range<Work>>>
  unitary_transform(InOutMat& A, const InMat& B, Work& C, M2&& map2, M3&& map3, P1&& proj1 = {}, P2&& proj2 = {}, P3&& proj3 = {}) {
    using std::begin, std::end; // for ADL
    std::fill(begin(C), end(C), 0);
    const auto map1 = mapping_transpose(map3);
    matrix_product(B, A, C, map1, map2, map2, proj1, proj2);
    std::fill(begin(A), end(A), 0);
    matrix_product(C, B, A, map2, map3, map2, identity, proj3);
  }

  /// @overload
  template <class InOutMat, class InMat, class M2, class M3, class P1 = conj_fn, class P2 = identity_fn, class P3 = identity_fn>
  std::enable_if_t<is_range_v<InOutMat> and is_range_v<InMat> and !is_range_v<M2>>
  unitary_transform(InOutMat& A, const InMat& B, M2&& map2, M3&& map3, P1&& proj1 = {}, P2&& proj2 = {}, P3&& proj3 = {}) {
    using T = remove_cvref_t<std::invoke_result_t<P2&, range_reference_t<InOutMat>>>;

    if constexpr (is_fixed_size_array_v<remove_cvref_t<InOutMat>>) {
      constexpr std::size_t N = fixed_size_matrix_dim_v<remove_cvref_t<InOutMat>>;
      static std::array<T, N * N> C;
      unitary_transform(A, B, C, map2, map3, proj1, proj2, proj3);
    } else {
      const std::size_t n = kspc::dim(A);
      std::vector<T> C(n * n);
      unitary_transform(A, B, C, map2, map3, proj1, proj2, proj3);
    }
  }

  /// @}
} // namespace kspc

// general matrix linear solve
namespace kspc {
  /// @addtogroup linalg
  /// @{

  /// @cond
  namespace detail {
    extern "C" {
      // double

      // LU factorization
      // http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60
      void dgetrf_(const std::size_t& m,
                  const std::size_t& n,
                  double* A, // M-by-N matrix
                  const std::size_t& lda,
                  std::size_t* ipiv,
                  int& info);

      // solve Ax = b with a general matrix A using LU factorization
      // http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga58e332cb1b8ab770270843221a48296d.html#ga58e332cb1b8ab770270843221a48296d
      void dgetrs_(const char& trans,
                  const std::size_t& n,
                  const std::size_t& nrhs,
                  const double* A, // N-by-N matrix
                  const std::size_t& lda,
                  const std::size_t* ipiv,
                  double* b, // N-by-NRHS matrix
                  const std::size_t& ldb,
                  int& info);

      // complex double

      // LU factorization
      void zgetrf_(const std::size_t& m,
                  const std::size_t& n,
                  std::complex<double>* A, // M-by-N matrix
                  const std::size_t& lda,
                  std::size_t* ipiv,
                  int& info);

      // solve Ax = b with a general matrix A using LU factorization
      void zgetrs_(const char& trans,
                  const std::size_t& n,
                  const std::size_t& nrhs,
                  const std::complex<double>* A, // N-by-N matrix
                  const std::size_t& lda,
                  const std::size_t* ipiv,
                  std::complex<double>* b, // N-by-NRHS matrix
                  const std::size_t& ldb,
                  int& info);
    }
  }
  /// @endcond

  /// LU factorization
  template <class InOutMat, class OutIPiv>
  int lu_factor(InOutMat& A, OutIPiv& ipiv) {
    using std::size, std::data; // for ADL
    const std::size_t n = kspc::dim(A);
    // std::cout << n << std::endl;
    assert(size(A) == n * n);
    assert(size(ipiv) == n);

    int info;
    if constexpr (is_complex_v<range_value_t<InOutMat>>) {
      //      zgetrf_(m, n,      A , lda,      ipiv , info)
      detail::zgetrf_(n, n, data(A),   n, data(ipiv), info);
    } else {
      //      dgetrf_(m, n,      A , lda,      ipiv , info)
      detail::dgetrf_(n, n, data(A),   n, data(ipiv), info);
    }
    return info;
  }

  /// solve Ax = b with a general matrix A using LU factorization
  template <class InMat, class InIPiv, class InOutVec>
  int matrix_vector_solve_with_lu_factor(const InMat& A, const InIPiv& ipiv, InOutVec& b) {
    using std::size, std::data; // for ADL
    const std::size_t n = kspc::dim(A);
    // std::cout << n << std::endl;
    assert(size(A) == n * n);
    assert(size(ipiv) == n);
    assert(size(b) == n);

    int info;
    if constexpr (is_complex_v<range_value_t<InMat>>) {
      //      zgetrs_(trans, n, nrhs,      A , lda,      ipiv ,      b , ldb, info)
      detail::zgetrs_(  'N', n,    1, data(A),   n, data(ipiv), data(b),   n, info);
    } else {
      //      dgetrs_(trans, n, nrhs,      A , lda,      ipiv ,      b , ldb, info)
      detail::dgetrs_(  'N', n,    1, data(A),   n, data(ipiv), data(b),   n, info);
    }
    return info;
  }

  /// solve Ax = b with a general matrix A
  template <class InOutMat, class OutIPiv, class InOutVec>
  std::enable_if_t<std::conjunction_v<
    is_range<InOutMat>, is_range<OutIPiv>, is_range<InOutVec>>, int>
  matrix_vector_solve(InOutMat& A, OutIPiv& ipiv, InOutVec& b) {
    int info;
    info = lu_factor(A, ipiv);
    if (info) return info;
    info = matrix_vector_solve_with_lu_factor(A, ipiv, b);
    return info;
  }

  /// @overload
  template <class InMat, class InOutVec, class M, class P = identity_fn>
  std::enable_if_t<
    is_range_v<InMat> and is_range_v<InOutVec> and (not is_range_v<M>) and (not is_range_v<P>), int>
  matrix_vector_solve(const InMat& A, InOutVec& b, M&& map, P&& proj = {}) {
    using T = remove_cvref_t<std::invoke_result_t<P&, range_reference_t<InMat>>>;
    int info;

    if constexpr (is_fixed_size_array_v<remove_cvref_t<InMat>>) {
      constexpr std::size_t N = fixed_size_matrix_dim_v<remove_cvref_t<InMat>>;
      static std::array<T, N * N> B;
      constexpr auto column_major = mapping_column_major(N);
      matrix_copy(A, B, map, column_major, proj);
      static std::array<std::size_t, N> ipiv;

      info = matrix_vector_solve(B, ipiv, b);
    } else {
      const std::size_t n = kspc::dim(A);
      std::vector<T> B(n * n);
      const auto column_major = mapping_column_major(n);
      matrix_copy(A, B, map, column_major, proj);
      std::vector<std::size_t> ipiv(n);

      info = matrix_vector_solve(B, ipiv, b);
    }

    return info;
  }

  /// @}
} // namespace kspc

// hermitian matrix eigen solve
namespace kspc::hermitian {
  /// @addtogroup linalg
  /// @{

  /// @cond
  namespace detail {
    extern "C" {
      // double

      // solve Ax = λx with a symmetric matrix A
      void dsyev_(const char& jobz,
                  const char& uplo,
                  const std::size_t& n,
                  double* A,
                  const std::size_t& lda,
                  double* w,
                  double* work,
                  const std::size_t& lwork,
                  int& info);

      // complex double

      // solve Ax = λx with a hermitian matrix A
      void zheev_(const char& jobz,
                  const char& uplo,
                  const std::size_t& n,
                  std::complex<double>* A,
                  const std::size_t& lda,
                  double* w,
                  std::complex<double>* work,
                  const std::size_t& lwork,
                  double* rwork,
                  int& info);
    }
  }
  /// @endcond

  /// solve Ax = λx with a symmetric matrix A
  template <class InOutMat, class OutVec, class Work>
  std::enable_if_t<std::conjunction_v<
    is_range<InOutMat>, is_range<OutVec>, is_range<Work>>, int>
  eigen_solve(InOutMat& A, OutVec& w, Work& work) {
    using std::size, std::data; // for ADL
    const std::size_t n = kspc::dim(A);
    // std::cout << n << std::endl;
    assert(size(A) == n * n);
    assert(size(w) == n);
    assert(size(work) >= 3 * n - 1);

    int info;
    //      dsyev_(jobz, uplo, n,      A , lda,      w ,      work ,     lwork , info)
    detail::dsyev_( 'V',  'U', n, data(A),   n, data(w), data(work), size(work), info);
    return info;
  }

  /// solve Ax = λx with a hermitian matrix A
  template <class InOutMat, class OutVec, class Work, class RWork>
  std::enable_if_t<std::conjunction_v<
    is_range<InOutMat>, is_range<OutVec>, is_range<Work>, is_range<RWork>>, int>
  eigen_solve(InOutMat& A, OutVec& w, Work& work, RWork& rwork) {
    using std::size, std::data; // for ADL
    const std::size_t n = kspc::dim(A);
    // std::cout << n << std::endl;
    assert(size(A) == n * n);
    assert(size(w) == n);
    assert(size(work) >= 2 * n - 1);
    assert(size(rwork) == 3 * n - 2);

    int info;
    //      zheev_(jobz, uplo, n,      A , lda,      w ,      work ,     lwork ,      rwork , info)
    detail::zheev_( 'V',  'U', n, data(A),   n, data(w), data(work), size(work), data(rwork), info);
    return info;
  }

  /// @overload
  template <class InOutMat, class OutVec, class M, class P = identity_fn>
  std::enable_if_t<
    is_range_v<InOutMat> and is_range_v<OutVec> and (not is_range_v<M>) and (not is_range_v<P>), int>
  eigen_solve(InOutMat& A, OutVec& w, M&& map, P&& proj = {}) {
    using T = remove_cvref_t<std::invoke_result_t<P&, range_reference_t<InOutMat>>>;
    int info;

    if constexpr (is_fixed_size_array_v<remove_cvref_t<InOutMat>>) {
      constexpr std::size_t N = fixed_size_matrix_dim_v<remove_cvref_t<InOutMat>>;
      static std::array<T, N * N> B;
      constexpr auto column_major = mapping_column_major(N);
      matrix_copy(A, B, map, column_major, proj);

      if constexpr (is_complex_v<T>) {
        static std::array<T, 4 * N> work;
        static std::array<complex_value_t<T>, N == 0 ? 0 : 3 * N - 2> rwork;
        info = eigen_solve(B, w, work, rwork);
      } else {
        static std::array<T, 6 * N> work;
        info = eigen_solve(B, w, work);
      }

      matrix_copy(B, A, column_major, map);
    } else {
      const std::size_t n = kspc::dim(A);
      std::vector<T> B(n * n);
      const auto column_major = mapping_column_major(n);
      matrix_copy(A, B, map, column_major, proj);

      if constexpr (is_complex_v<T>) {
        std::vector<T> work(4 * n);
        std::vector<complex_value_t<T>> rwork(n == 0 ? 0 : 3 * n - 2);
        info = eigen_solve(B, w, work, rwork);
      } else {
        std::vector<T> work(6 * n);
        info = eigen_solve(B, w, work);
      }

      matrix_copy(B, A, column_major, map);
    }

    return info;
  }

  /// @}
} // namespace kspc

// clang-format on
