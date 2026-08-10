//
// Created by Osayamen on 6/2/26.
//
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
  const std::vector<std::uintptr_t>& var_signal_table,
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
  void* varLenSignals = nullptr;
  void* varOffsetSignals = nullptr;

  CHECK_CUDA(cudaMallocAsync(&stagingTR, sizeof(cuda::std::byte*) * world, stream));
  static_assert(sizeof(uintptr_t) == sizeof(cuda::std::byte*));
  CHECK_CUDA(cudaMemcpyAsync(stagingTR, staging_table.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMallocAsync(&stagingLR, sizeof(cuda::std::byte*) * world, stream));
  CHECK_CUDA(cudaMallocAsync(&signals, sizeof(uint64_t*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(signals, signal_table.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMallocAsync(&gatherSignals, sizeof(uint64_t*) * world, stream));
  CHECK_CUDA(cudaMallocAsync(&varLenSignals, sizeof(purlin::LRP*) * world, stream));
  CHECK_CUDA(cudaMemcpyAsync(varLenSignals, var_signal_table.data(), sizeof(purlin::LRP*) * world,
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMallocAsync(&varOffsetSignals, sizeof(purlin::LRP*) * world, stream));

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

  std::vector<uintptr_t> varSigStash(world);
  const auto offsetVarSig = 2 * world;
  for (int i = 0; i < world; i ++) {
    auto* __restrict__ p = reinterpret_cast<purlin::LRP*>(var_signal_table[i]);
    if (!cuda::is_aligned(p, sizeof(purlin::LRP))) {
      throw std::runtime_error("var-len signal is not aligned to at least 16 bytes");
    }
    varSigStash[i] = reinterpret_cast<uintptr_t>(p + offsetVarSig);
  }
  CHECK_CUDA(cudaMemcpyAsync(varOffsetSignals, varSigStash.data(), sizeof(uintptr_t) * world,
    cudaMemcpyHostToDevice, stream));

  const auto ctx = purlin::initialize(rank, world,
    static_cast<cuda::std::byte**>(stagingLR),
    static_cast<cuda::std::byte**>(stagingTR),
    static_cast<uint64_t**>(signals),
    static_cast<uint64_t**>(gatherSignals),
    static_cast<purlin::LRP**>(varLenSignals),
    static_cast<purlin::LRP**>(varOffsetSignals),
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
  CHECK_CUDA(cudaFreeAsync(ctx->varLenSignals, stream));
  CHECK_CUDA(cudaFreeAsync(ctx->varOffsetSignals, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  delete ctx;
}

static void all_gather(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  purlin::allGather<ARCH>(
    reinterpret_cast<cuda::std::byte*>(src),
    reinterpret_cast<cuda::std::byte*>(dst),
    bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
}

static void all_gather_v(const uintptr_t& src, const uintptr_t& dst,
  const std::vector<size_t>& sizes,
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto ctx = *reinterpret_cast<purlin::Context*>(raw_ctx);
  CHECK_CUDA(cudaMemcpyAsync(ctx.sizes,
    sizes.data(),
    sizes.size() * sizeof(size_t),
    cudaMemcpyHostToDevice, stream));
  purlin::VState vState{};
  vState.maxBytes = 0;
  vState.totalBytes = 0;
  vState.bytes = sizes[ctx.rank];
  vState.offset = 0;
  for (int i = 0; i < sizes.size(); ++i) {
    const auto size = sizes[i];
    if (size % purlin::MAX_ACCESS_ALIGNMENT != 0) {
      throw std::runtime_error("Size is invalid");
    }
    vState.maxBytes = vState.maxBytes >= size ? vState.maxBytes : size;
    vState.totalBytes += size;
    if (i < ctx.rank) {
      vState.offset += size;
    }
  }
  ctx.vState = vState;
  purlin::allGatherV<ARCH>(reinterpret_cast<cuda::std::byte*>(src), reinterpret_cast<cuda::std::byte*>(dst),
    ctx.sizes, ctx, stream);
}

static void all_reduce(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const int& buffer_type, const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  switch (buffer_type) {
    case purlin::TensorType::fp16: {
      purlin::allReduce<ARCH, __half>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    case purlin::TensorType::bf16: {
      purlin::allReduce<ARCH, __nv_bfloat16>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    case purlin::TensorType::fp8E4M3: {
      purlin::allReduce<ARCH, __nv_fp8_e4m3>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    case purlin::TensorType::fp8E5M2: {
      purlin::allReduce<ARCH, __nv_fp8_e5m2>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst),
        bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
      break;
    default: {
      purlin::allReduce<ARCH, float>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst),
      bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
    }
  }
}

static void all_to_all(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  purlin::all2all<ARCH>(
    reinterpret_cast<cuda::std::byte*>(src),
    reinterpret_cast<cuda::std::byte*>(dst),
    bytes, *reinterpret_cast<purlin::Context*>(raw_ctx), reinterpret_cast<cudaStream_t>(stream_ptr));
}

static void all_to_all_v(const uintptr_t& src, const uintptr_t& dst,
  const std::vector<size_t>&& splits, // [in_splits, out_splits]
  const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto ctx = *reinterpret_cast<purlin::Context*>(raw_ctx);
  CHECK_CUDA(cudaMemcpyAsync(ctx.sizes,
    splits.data(),
    splits.size() * sizeof(size_t),
    cudaMemcpyHostToDevice, stream));
  purlin::VState vState{};
  vState.maxBytes = 0;
  vState.totalBytes = 0;
  vState.maxOutBytes = 0;
  vState.totalOutBytes = 0;
  const auto* __restrict__ in_splits = splits.data();
  const auto* __restrict__ out_splits = splits.data() + ctx.world;
  for (int i = 0; i < ctx.world; ++i) {
    const auto size = in_splits[i];
    const auto outSize = out_splits[i];
    if (size % purlin::MAX_ACCESS_ALIGNMENT != 0) {
      throw std::runtime_error("Size is invalid");
    }
    vState.maxBytes = vState.maxBytes >= size ? vState.maxBytes : size;
    vState.maxOutBytes = vState.maxOutBytes >= outSize ? vState.maxOutBytes : outSize;
    vState.totalBytes += size;
    vState.maxOutBytes += size;
  }
  ctx.vState = vState;
  purlin::all2allV<ARCH>(reinterpret_cast<cuda::std::byte*>(src), reinterpret_cast<cuda::std::byte*>(dst),
    ctx.sizes, ctx.sizes + ctx.world, ctx, stream);
}

static void reduce_scatter(const uintptr_t& src, const uintptr_t& dst, const size_t& bytes,
  const int& buffer_type, const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  const auto ctx = *reinterpret_cast<purlin::Context*>(raw_ctx);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto localBytes = bytes / ctx.world_l;
  switch (buffer_type) {
    case purlin::TensorType::fp16: {
      purlin::reduceScatter<ARCH, __half>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    case purlin::TensorType::bf16: {
      purlin::reduceScatter<ARCH, __nv_bfloat16>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    case purlin::TensorType::fp8E4M3: {
      purlin::reduceScatter<ARCH, __nv_fp8_e4m3>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    case purlin::TensorType::fp8E5M2: {
      purlin::reduceScatter<ARCH, __nv_fp8_e5m2>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
      break;
    default: {
      purlin::reduceScatter<ARCH, float>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst), localBytes, ctx, stream);
    }
  }
}

static void reduce_scatter_v(const uintptr_t& src, const uintptr_t& dst,
  const std::vector<size_t>& sizes,
  const int& buffer_type, const uintptr_t& raw_ctx, const uintptr_t& stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto ctx = *reinterpret_cast<purlin::Context*>(raw_ctx);
  CHECK_CUDA(cudaMemcpyAsync(ctx.sizes,
    sizes.data(),
    sizes.size() * sizeof(size_t),
    cudaMemcpyHostToDevice, stream));
  purlin::VState vState{};
  vState.maxBytes = 0;
  vState.totalBytes = 0;
  vState.bytes = sizes[ctx.rank];
  vState.offset = 0;
  for (int i = 0; i < sizes.size(); ++i) {
    const auto size = sizes[i];
    if (size % purlin::MAX_ACCESS_ALIGNMENT != 0) {
      throw std::runtime_error("Size is invalid");
    }
    vState.maxBytes = vState.maxBytes >= size ? vState.maxBytes : size;
    vState.totalBytes += size;
    if (i < ctx.rank) {
      vState.offset += size;
    }
  }
  ctx.vState = vState;
  switch (buffer_type) {
    case purlin::TensorType::fp16: {
      purlin::reduceScatterV<ARCH, __half>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), ctx.sizes, ctx, stream);
    }
      break;
    case purlin::TensorType::bf16: {
      purlin::reduceScatterV<ARCH, __nv_bfloat16>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), ctx.sizes, ctx, stream);
    }
      break;
    case purlin::TensorType::fp8E4M3: {
      purlin::reduceScatterV<ARCH, __nv_fp8_e4m3>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), ctx.sizes, ctx, stream);
    }
      break;
    case purlin::TensorType::fp8E5M2: {
      purlin::reduceScatterV<ARCH, __nv_fp8_e5m2>(reinterpret_cast<cuda::std::byte*>(src),
        reinterpret_cast<cuda::std::byte*>(dst), ctx.sizes, ctx, stream);
    }
      break;
    default: {
      purlin::reduceScatterV<ARCH, float>(reinterpret_cast<cuda::std::byte*>(src),
      reinterpret_cast<cuda::std::byte*>(dst), ctx.sizes, ctx, stream);
    }
  }
}

PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &purlin_initialize);
  m.def("finalize", &purlin_finalize);
  m.def("all_gather", &all_gather);
  m.def("all_gather_v", &all_gather_v);
  m.def("all_reduce", &all_reduce);
  m.def("all_to_all", &all_to_all);
  m.def("all_to_all_v", &all_to_all_v);
  m.def("reduce_scatter", &reduce_scatter);
  m.def("reduce_scatter_v", &reduce_scatter_v);
}
