//
// Created by Osayamen on 6/2/26.
//
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <purlin/host.cuh>

namespace py = pybind11;

static std::uintptr_t purlin_initialize(const int& rank,
  const int& world,
  const std::vector<std::uintptr_t>& staging_table,
  const std::vector<std::uintptr_t>& signal_table,
  const uint64_t& staging_size,
  const std::uintptr_t& stream_ptr) {
  if (staging_size > purlin::MAX_STAGING_SIZE || staging_size < purlin::MIN_CHUNK_SIZE) {
    throw std::runtime_error("staging size is invalid");
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  // allocate pointer tables
  void* stagingTR = nullptr;
  void* stagingLR = nullptr;
  void* signals = nullptr;
  void* gatherSignals = nullptr;

  CHECK_CUDA(cudaMallocAsync(&stagingTR, sizeof(cuda::std::byte*) * world, stream));
  static_assert(sizeof(uintptr_t) == sizeof(cuda::std::byte*));
  CHECK_CUDA(cudaMemcpyAsync(stagingTR, staging_table.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMallocAsync(&stagingLR, sizeof(cuda::std::byte*) * world, stream));
  CHECK_CUDA(cudaMallocAsync(&signals, sizeof(uint64_t*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(signals, signal_table.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMallocAsync(&gatherSignals, sizeof(uint64_t*) * world, stream));

  std::vector<uintptr_t> stagingStash(world);
  const auto offsetTR = 2 * staging_size;
  for (int i = 0; i < world; i++) {
    stagingStash[i] = reinterpret_cast<uintptr_t>(reinterpret_cast<cuda::std::byte*>(staging_table[i]) + offsetTR);
  }
  CHECK_CUDA(cudaMemcpyAsync(stagingLR, stagingStash.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream));
  std::vector<uintptr_t> signalStash(world);
  const auto offsetSig = world;
  for (int i = 0; i < world; i++) {
    signalStash[i] = reinterpret_cast<uintptr_t>(reinterpret_cast<uint64_t*>(signal_table[i]) + offsetSig);
  }
  CHECK_CUDA(cudaMemcpyAsync(gatherSignals, signalStash.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream));

  const auto ctx = purlin::initialize(rank, world,
    static_cast<cuda::std::byte**>(stagingLR),
    static_cast<cuda::std::byte**>(stagingTR),
    static_cast<uint64_t**>(signals),
    static_cast<uint64_t**>(gatherSignals),
    staging_size,
    stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto* pyCtx = new purlin::Context(ctx);
  return reinterpret_cast<uintptr_t>(pyCtx);
}

static void purlin_finalize(const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto* ctx = reinterpret_cast<purlin::Context*>(raw_ctx);
  if (!ctx) return;
  purlin::finalize(*ctx, stream);
  CHECK_CUDA(cudaFreeAsync(ctx->staging, stream));
  CHECK_CUDA(cudaFreeAsync(ctx->stagingLR, stream));
  CHECK_CUDA(cudaFreeAsync(ctx->signals, stream));
  CHECK_CUDA(cudaFreeAsync(ctx->gatherSignals, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  delete ctx;
}

static void all_gather(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  purlin::allGather(
    reinterpret_cast<cuda::std::byte*>(src),
    reinterpret_cast<cuda::std::byte*>(dst),
    bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
}

static void all_reduce(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const int& buffer_type, const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  switch (buffer_type) {
    case purlin::TensorType::fp16: {
      purlin::allReduce<__half>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    case purlin::TensorType::bf16: {
      purlin::allReduce<__nv_bfloat16>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    case purlin::TensorType::fp32: {
      purlin::allReduce<float>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst),
      bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    default: {
      // fp64
      purlin::allReduce<double>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst),
      bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
  }
}

static void all_to_all(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  purlin::all2all(
    reinterpret_cast<cuda::std::byte*>(src),
    reinterpret_cast<cuda::std::byte*>(dst),
    bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
}

static void reduce_scatter(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const int& buffer_type, const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  const auto ctx = *reinterpret_cast<purlin::Context*>(raw_ctx);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto localBytes = bytes / ctx.world_l;
  switch (buffer_type) {
    case purlin::TensorType::fp16: {
      purlin::reduceScatter<__half>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    case purlin::TensorType::bf16: {
      purlin::reduceScatter<__nv_bfloat16>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    case purlin::TensorType::fp32: {
      purlin::reduceScatter<float>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    default: {
      // fp64
      purlin::reduceScatter<double>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
  }
}

PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &purlin_initialize);
  m.def("finalize", &purlin_finalize);
  m.def("all_gather", &all_gather);
  m.def("all_reduce", &all_reduce);
  m.def("all_to_all", &all_to_all);
  m.def("reduce_scatter", &reduce_scatter);
}

int main() {

}
