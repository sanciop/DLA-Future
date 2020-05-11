//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a) noexcept {
  DLAF_ASSERT((a.size().rows() == a.size().cols()), "a is not square (", a.size().rows(),
              " != ", a.size().cols(), ")");

  auto info = lapack::potrf(uplo, a.size().rows(), a.ptr(), a.ld());
  DLAF_ASSERT_HEAVY((info >= 0));

  return info;
}

template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a) noexcept {
  auto info = potrfInfo(uplo, a);

  DLAF_ASSERT((info == 0), "a is not positive definite");
}
