#ifndef PURLIN_SUPPORT_BENCHMARK_KERNEL_CUH
#define PURLIN_SUPPORT_BENCHMARK_KERNEL_CUH

#include "benchmark.cuh"

namespace bench {

struct KernelOptions : Options {
  int maxBlocks = -1;
};

// Device-kernel examples insert a block limit before the standard timing options:
// [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed].
inline KernelOptions parseKernelOptions(int argc, char** argv, const int defaultMaxBlocks = -1) {
  KernelOptions options;
  options.maxBlocks = defaultMaxBlocks;
  std::vector<char*> arguments{argv[0]};
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--reduction-mode") {
      arguments.push_back(argv[i]);
      if (i + 1 < argc) arguments.push_back(argv[++i]);
    } else if (++positional == 3) {
      options.maxBlocks = std::stoi(argv[i]);
    } else {
      arguments.push_back(argv[i]);
    }
  }
  static_cast<Options&>(options) = parseOptions(static_cast<int>(arguments.size()), arguments.data());
  return options;
}

template<typename Kernel>
inline void configureKernel(Kernel kernel, const size_t sharedBytes, const cudaDeviceProp& device) {
  if (sharedBytes > device.sharedMemPerBlockOptin) {
    throw std::runtime_error("Required shared memory " + std::to_string(sharedBytes) +
      " exceeds hardware limits: " + std::to_string(device.sharedMemPerBlockOptin));
  }
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(sharedBytes)));
}

} // namespace bench

#endif
