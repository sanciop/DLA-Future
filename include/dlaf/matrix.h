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
#include <exception>
#include <vector>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/internal/tile_future_manager.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
class MatrixView;

/// A @c Matrix object represents a collection of tiles which contain all the elements of a matrix.
///
/// The tiles are distributed according to a distribution (see @c Matrix::distribution()),
/// therefore some tiles are stored locally on this rank,
/// while the others are available on other ranks.
/// More details are available in misc/matrix_distribution.md.
/// TODO: Sync details.

template <class T, Device device>
class Matrix : public Matrix<const T, device> {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;
  friend Matrix<const ElementType, device>;
  friend MatrixView<ElementType, device>;

  /// Create a non distributed matrix of size @p size and block size @p block_size
  ///
  /// @throw std::invalid_argument if @p !size.isValid().
  /// @throw std::invalid_argument if @p !block_size.isValid() or @p block_size_.isEmpty().
  Matrix(const LocalElementSize& size, const TileElementSize& block_size);

  /// Create a distributed matrix of size @p size and block size @p block_size on the given 2D
  /// communicator grid @p comm
  ///
  /// @throw std::invalid_argument if @p !size.isValid().
  /// @throw std::invalid_argument if @p !block_size.isValid() or @p block_size_.isEmpty().
  Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
         const comm::CommunicatorGrid& comm);

  /// Create a matrix distributed according to the distribution @p distribution.
  Matrix(Distribution&& distribution);

  /// Create a matrix distributed according to the distribution @p distribution,
  /// specifying the layout.
  ///
  /// @param[in] layout is the layout which describes how the elements
  ///            of the local part of the matrix will be stored in memory.
  Matrix(Distribution&& distribution, const LayoutInfo& layout);

  /// Create a non distributed matrix,
  /// which references elements that are already allocated in the memory
  ///
  /// @param[in] layout is the layout which describes how the elements
  ///            of the local part of the matrix are stored in memory.
  /// @param[in] ptr is the pointer to the first element of the local part of the matrix.
  /// @pre @p ptr refers to an allocated memory region of at least @c layout.minMemSize() elements.
  Matrix(const LayoutInfo& layout, ElementType* ptr);

  /// Create a matrix distributed according to the distribution @p distribution,
  /// which references elements that are already allocated in the memory
  ///
  /// @param[in] layout is the layout which describes how the elements
  ///            of the local part of the matrix are stored in memory.
  /// @param[in] ptr is the pointer to the first element of the local part of the matrix.
  /// @throw std::invalid_argument if @p distribution.localSize() != @p layout.size().
  /// @throw std::invalid_argument if @p distribution.blockSize() != @p layout.blockSize().
  /// @pre @p ptr refers to an allocated memory region of at least @c layout.minMemSize() elements.
  Matrix(Distribution&& distribution, const LayoutInfo& layout, ElementType* ptr);

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// @brief Returns a future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::future<TileType> operator()(const LocalTileIndex& index) noexcept;

  /// @brief Returns a future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throw std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::future<TileType> operator()(const GlobalTileIndex& index) {
    return operator()(this->distribution().localTileIndex(index));
  }

private:
  using Matrix<const T, device>::tileManager;
  using Matrix<const T, device>::setUpTiles;
};

template <class T, Device device>
class Matrix<const T, device> : public internal::MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;
  friend Matrix<ElementType, device>;

  Matrix(const LayoutInfo& layout, ElementType* ptr);

  Matrix(const LayoutInfo& layout, const ElementType* ptr)
      : Matrix(layout, const_cast<ElementType*>(ptr)) {}

  Matrix(Distribution&& distribution, const LayoutInfo& layout, ElementType* ptr);

  Matrix(Distribution&& distribution, const LayoutInfo& layout, const ElementType* ptr)
      : Matrix(std::move(distribution), layout, const_cast<ElementType*>(ptr)) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  virtual ~Matrix();

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throw std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const GlobalTileIndex& index) {
    return read(distribution().localTileIndex(index));
  }

private:
  Matrix(Distribution&& distribution,
         std::vector<internal::TileFutureManager<T, device>>&& tile_managers);

  void setUpTiles(const memory::MemoryView<ElementType, device>& mem, const LayoutInfo& layout) noexcept;

  internal::TileFutureManager<T, device>& tileManager(const LocalTileIndex& index);

  std::vector<internal::TileFutureManager<T, device>> tile_managers_;
};

// Note: the templates of the following helper functions are inverted w.r.t. the Matrix templates
// to allow the user to only specify the device and let the compiler deduce the type T.

// Local versions

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a column major layout
///
/// @param[in] ld the leading dimension of the matrix.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld < max(1, size.row()).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const LocalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld, T* ptr) {
  return Matrix<T, device>(colMajorLayout(size, block_size, ld), ptr);
}

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a tile layout
///
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                       T* ptr) {
  return Matrix<T, device>(tileLayout(size, block_size), ptr);
}

/// Create a non distributed matrix of size @p size and block size @p block_size
/// which references elements
/// that are already allocated in the memory with a tile layout
///
/// @param[in] ld_tile the leading dimension of the tiles.
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld_tile < max(1, min(block_size.row(), size.row())).
/// @throw std::invalid_argument if @p tiles_per_col < ceilDiv(size.row(), block_size.col()).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col, T* ptr) {
  return Matrix<T, device>(tileLayout(size, block_size, ld_tile, tiles_per_col), ptr);
}

// Distributed versions

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a column major layout
///
/// @param[in] ld the leading dimension of the matrix.
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld < max(1, size.row()).
/// @throw std::invalid_argument if @p !source_rank_index.isValid() or @p !source_rank_index_.isIn(grid_size).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld,
                                           const comm::CommunicatorGrid& comm,
                                           const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = colMajorLayout(distribution.localSize(), block_size, ld);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a column major layout
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ld the leading dimension of the matrix.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld < max(1, size.row()).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld,
                                           const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromColMajor<device>(size, block_size, ld, comm, {0, 0}, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout
///
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p !source_rank_index.isValid() or @p !source_rank_index_.isIn(grid_size).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       const comm::CommunicatorGrid& comm,
                                       const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = tileLayout(distribution.localSize(), block_size);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<device>(size, block_size, comm, {0, 0}, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout
///
/// @param[in] ld_tile the leading dimension of the tiles.
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles.
/// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld_tile < max(1, min(block_size.row(), size.row())).
/// @throw std::invalid_argument if @p tiles_per_col < ceilDiv(size.row(), block_size.row()).
/// @throw std::invalid_argument if @p !source_rank_index.isValid() or @p !source_rank_index_.isIn(grid_size).

/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col,
                                       const comm::CommunicatorGrid& comm,
                                       const comm::Index2D& source_rank_index, T* ptr) {
  Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = tileLayout(distribution.localSize(), block_size, ld_tile, tiles_per_col);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

/// Create a distributed matrix of size @p size and block size @p block_size
/// on the given 2D communicator grid @p comm which references elements
/// that are already allocated in the memory with a tile layout
///
/// This method assumes @p source_rank_index to be {0,0}.
/// @param[in] ld_tile the leading dimension of the tiles.
/// @param[in] tiles_per_col the number of tiles stored for each column of tiles.
/// @param[in] ptr is the pointer to the first element of the local part of the matrix.
/// @throw std::invalid_argument if @p ld_tile < max(1, min(block_size.row(), size.row()).
/// @throw std::invalid_argument if @p tiles_per_col < ceilDiv(size.row(), block_size.col()).
/// @pre @p ptr refers to an allocated memory region which can contain the elements of the local matrix
/// stored in the given layout.
template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col,
                                       const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<device>(size, block_size, ld_tile, tiles_per_col, comm, {0, 0}, ptr);
}

/// ---- ETI

#define DLAF_MATRIX_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class Matrix<DATATYPE, DEVICE>; \
  KWORD template class Matrix<const DATATYPE, DEVICE>;

DLAF_MATRIX_ETI(extern, float, Device::CPU)
DLAF_MATRIX_ETI(extern, double, Device::CPU)
DLAF_MATRIX_ETI(extern, std::complex<float>, Device::CPU)
DLAF_MATRIX_ETI(extern, std::complex<double>, Device::CPU)

// DLAF_MATRIX_ETI(extern, float, Device::GPU)
// DLAF_MATRIX_ETI(extern, double, Device::GPU)
// DLAF_MATRIX_ETI(extern, std::complex<float>, Device::GPU)
// DLAF_MATRIX_ETI(extern, std::complex<double>, Device::GPU)

}
/// Make dlaf::matrix::Matrix available as dlaf::Matrix.
using matrix::Matrix;
}

#include "dlaf/matrix.tpp"
#include "dlaf/matrix_const.tpp"
