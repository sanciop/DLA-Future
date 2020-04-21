#include <chrono>
#include <hpx/hpx.hpp>

int main_hpx(int argc, char** argv) {
  hpx::shared_future<void> shared_future = hpx::make_ready_future();

  hpx::future<void> last_task =
      shared_future.then([](auto&&) { hpx::this_thread::sleep_for(std::chrono::milliseconds(10)); });

  last_task.get();

  return hpx::finalize();
}

int main(int argc, char** argv) {
  return hpx::init(main_hpx, argc, argv);
}
