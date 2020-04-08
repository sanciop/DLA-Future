//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <mpi.h>
#include <hpx/hpx_init.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/factorization/mc.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/common/timer.h"

int hpx_main(hpx::program_options::variables_map&) {
  // TODO: CUDA gemm
  return hpx::finalize();
}

int main(int argc, char** argv) {
  // Initialize MPI
  int threading_required = MPI_THREAD_MULTIPLE;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  auto ret_code = hpx::init(hpx_main, argc, argv);

  MPI_Finalize();
  return ret_code;
}
