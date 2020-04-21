#include <exception>

#include <hpx/hpx.hpp>
#include <stdexcept>

#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"

using T = std::complex<float>;  // randomly chosen element type for matrix

using dlaf::Matrix;
using dlaf::LocalTileIndex;
using dlaf::matrix::LayoutInfo;
using dlaf::memory::MemoryView;

const auto WAIT_GUARD = std::chrono::milliseconds(10);
const auto device = dlaf::Device::CPU;

template <class T>
auto createMatrix() -> Matrix<T, device> {
  return {{1, 1}, {1, 1}};
}

template <class T>
auto createConstMatrix() {
  LayoutInfo layout({1, 1}, {1, 1}, 1, 1, 1);
  MemoryView<T, device> mem(layout.minMemSize());
  const T* p = mem();

  return Matrix<const T, device>{layout, p};
}

// NonConstAfterRead
void test_me() {
  hpx::future<void> last_task;

  volatile int guard = 0;
  {
    auto matrix = createMatrix<T>();

    auto shared_future = matrix.read(LocalTileIndex(0, 0));
    last_task = shared_future.then(hpx::launch::async, [&guard](auto&&) {
      hpx::this_thread::sleep_for(WAIT_GUARD);
      if (guard != 0)
        throw std::runtime_error("OPPALA");
    });
  }
  guard = 1;

  last_task.get();
}

int main_hpx(int argc, char **argv) {
  test_me();
  return hpx::finalize();
}

int main(int argc, char **argv) {
  auto ret_code = hpx::init(main_hpx, argc, argv);

  return ret_code;
}
