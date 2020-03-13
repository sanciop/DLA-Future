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
Matrix<const T, device>::Matrix(const LayoutInfo& layout, ElementType* ptr)
    : MatrixBase({layout.size(), layout.blockSize()}) {
  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<const T, device>::Matrix(Distribution&& distribution, const LayoutInfo& layout, ElementType* ptr)
    : MatrixBase(std::move(distribution)) {
  if (this->distribution().localSize() != layout.size())
    throw std::invalid_argument("Error: distribution.localSize() != layout.size()");
  if (this->blockSize() != layout.blockSize())
    throw std::invalid_argument("Error: distribution.blockSize() != layout.blockSize()");

  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<const T, device>::~Matrix() {
  for (auto&& tile_manager : tile_managers_) {
    try {
      tile_manager.clearSync();
    }
    catch (...) {
      // TODO WARNING
    }
  }
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> Matrix<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  return tileManager(index).getReadTileSharedFuture();
}

template <class T, Device device>
Matrix<const T, device>::Matrix(Distribution&& distribution,
                                std::vector<internal::TileFutureManager<T, device>>&& tile_managers)

    : MatrixBase(std::move(distribution)), tile_managers_(std::move(tile_managers)) {}

template <class T, Device device>
void Matrix<const T, device>::setUpTiles(const memory::MemoryView<ElementType, device>& mem,
                                         const LayoutInfo& layout) noexcept {
  const auto& nr_tiles = layout.nrTiles();

  tile_managers_.clear();
  tile_managers_.reserve(futureVectorSize(nr_tiles));

  using MemView = memory::MemoryView<T, device>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_managers_.emplace_back(
          TileType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                   layout.ldTile()));
    }
  }
}

template <class T, Device device>
internal::TileFutureManager<T, device>& Matrix<const T, device>::tileManager(
    const LocalTileIndex& index) {
  std::size_t i = tileLinearIndex(index);
  return tile_managers_[i];
}

}
}
