//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include "lapack.hh"
// LAPACKPP includes complex.h which defines the macro I.
// This breaks HPX.
#ifdef I
#undef I
#endif

#include "dlaf/tile.h"

namespace dlaf {
namespace tile {

// See LAPACK documentation for more details.

// Variants that throw an error on failure.

/// Compute the cholesky decomposition of a.

/// Only the upper or lower triangular elements are referenced according to @p uplo.
///
/// When the assertion is enabled, terminates the program with an error message
/// if matrix is not square or when it is not positive definite.
/// This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a);

// Variants that return info code.

/// Compute the cholesky decomposition of a.

/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
/// When the assertion is enabled, terminates the program with an error message
/// if matrix is not positive definite.
/// This assertion is enabled when **DLAF_ASSERT_ENABLE** is ON.
template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a);

#include "dlaf/lapack_tile.tpp"
}
}
