//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device device>
template <template <class, Device> class MatrixType>
MatrixView<T, device>::MatrixView(blas::Uplo uplo, MatrixType<T, device>& matrix, bool force_RW)
    : MatrixBase(matrix) {
  if (uplo != blas::Uplo::General)
    throw std::invalid_argument("uplo != General not implemented yet.");
  setUpTiles(matrix, force_RW);
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> MatrixView<T, device>::read(
    const LocalTileIndex& index) noexcept {
  return tileManager(index).getReadTileSharedFuture();
}

template <class T, Device device>
hpx::future<Tile<T, device>> MatrixView<T, device>::operator()(const LocalTileIndex& index) noexcept {
  return tileManager(index).getRWTileFuture();
}

template <class T, Device device>
void MatrixView<T, device>::doneWrite(const LocalTileIndex& index) noexcept {
  return tileManager(index).makeRead();
}

template <class T, Device device>
void MatrixView<T, device>::done(const LocalTileIndex& index) noexcept {
  return tileManager(index).release();
}

template <class T, Device device>
template <template <class, Device> class MatrixType>
void MatrixView<T, device>::setUpTiles(MatrixType<T, device>& matrix, bool force_RW) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_managers_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      tile_managers_.emplace_back(matrix.tileManager(ind), force_RW);
    }
  }
}

template <class T, Device device>
internal::ViewTileFutureManager<T, device>& MatrixView<T, device>::tileManager(
    const LocalTileIndex& index) {
  std::size_t i = tileLinearIndex(index);
  return tile_managers_[i];
}

}
}
