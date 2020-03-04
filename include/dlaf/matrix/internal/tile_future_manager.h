//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <hpx/hpx.hpp>
#include "dlaf/tile.h"

namespace dlaf {
namespace matrix {
namespace internal {

template <class T, Device device>
struct TileFutureManager {
  TileFutureManager() {}

  TileFutureManager(Tile<T, device>&& tile) : tile_future_(hpx::make_ready_future(std::move(tile))) {}

  // The future of the tile with no promise set.
  hpx::future<Tile<T, device>> tile_future_;

  // If valid, a copy of the shared future of the tile,
  // which has the promise set to the promise for tile_future_.
  hpx::shared_future<Tile<const T, device>> tile_shared_future_;
};

template <class ReturnTileType, class TileType>
hpx::future<ReturnTileType> setPromiseTileFuture(hpx::future<TileType>& old_future,
                                                 hpx::promise<TileType>& p) {
  return old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
    try {
      return ReturnTileType(std::move(fut.get().setPromise(std::move(p))));
    }
    catch (...) {
      auto current_exception_ptr = std::current_exception();
      p.set_exception(current_exception_ptr);
      std::rethrow_exception(current_exception_ptr);
    }
  });
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> getReadTileSharedFuture(
    TileFutureManager<T, device>& tile_manager) noexcept {
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;

  if (!tile_manager.tile_shared_future_.valid()) {
    hpx::future<TileType> old_future = std::move(tile_manager.tile_future_);
    hpx::promise<TileType> p;
    tile_manager.tile_future_ = p.get_future();
    tile_manager.tile_shared_future_ = std::move(setPromiseTileFuture<ConstTileType>(old_future, p));
  }
  return tile_manager.tile_shared_future_;
}

template <class T, Device device>
hpx::future<Tile<T, device>> getRWTileFuture(TileFutureManager<T, device>& tile_manager) noexcept {
  using TileType = Tile<T, device>;

  hpx::future<TileType> old_future = std::move(tile_manager.tile_future_);
  hpx::promise<TileType> p;
  tile_manager.tile_future_ = p.get_future();
  tile_manager.tile_shared_future_ = {};
  return setPromiseTileFuture<TileType>(old_future, p);
}

}
}
}
