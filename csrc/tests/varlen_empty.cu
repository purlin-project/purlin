#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/host/allGather.cuh>
#include <purlin/host/all2all.cuh>
#include <purlin/host/reduceScatter.cuh>

namespace {
using Element = __nv_bfloat16;
enum class Operation { agv, rsv, a2av };
enum class Pattern { oneWay, selfOnly, mixedRing };

__global__ void delayRank() {
  const auto start = clock64();
  while (clock64() - start < 20000000ULL) {}
}

__global__ void advanceIteration(unsigned* iteration) { ++*iteration; }

// Small integers keep the reductions exact; source, destination, element and
// invocation all affect the data. The device counter also changes on replay.
__device__ Element sample(int source, int destination, size_t index, unsigned iteration) {
  return __float2bfloat16_rn(static_cast<float>(
    1 + (source * 3 + destination * 5 + index + iteration) % 16));
}

__global__ void fill(Element* data, size_t count, int source, int destination,
                     const unsigned* iteration) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    data[i] = sample(source, destination, i, *iteration);
  }
}

__global__ void check(const Element* data, size_t count, int source, int destination,
                      int world, bool reduce, const unsigned* iteration, int* errors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    Element expected = sample(source, destination, i, *iteration);
    if (reduce) {
      expected = sample(0, destination, i, *iteration);
      for (int peer = 1; peer < world; ++peer) {
        expected = __float2bfloat16_rn(__bfloat162float(expected) +
          __bfloat162float(sample(peer, destination, i, *iteration)));
      }
    }
    if (__bfloat162float(data[i]) != __bfloat162float(expected)) atomicAdd(errors, 1);
  }
}

struct Layout {
  std::vector<size_t> sends, receives;
};

Layout makeLayout(Operation op, Pattern pattern, size_t total, int owner, int rank, int world) {
  Layout layout{std::vector<size_t>(world, 0), std::vector<size_t>(world, 0)};
  if (op == Operation::agv) {
    layout.sends[rank] = rank == owner ? total : 0;
    layout.receives[owner] = total;
  } else if (op == Operation::rsv) {
    layout.sends[owner] = total;
    layout.receives[rank] = rank == owner ? total : 0;
  } else {
    // One-way traffic leaves empty inputs/outputs. Self-only traffic also
    // leaves fully idle ranks. The ring mixes packet and chunked streams with
    // different local extents, exercising A2AV's deferred global epoch advance.
    for (int source = 0; source < world; ++source) {
      const int destination = pattern == Pattern::selfOnly ? source : (source + 1) % world;
      const size_t bytes = source == owner ? total : (pattern == Pattern::mixedRing ? 128 : 0);
      if (source == rank) layout.sends[destination] = bytes;
      if (destination == rank) layout.receives[source] = bytes;
    }
  }
  return layout;
}

struct Case {
  Operation op;
  const char* name;
  size_t total;
  Layout layout;
  std::vector<size_t> sendOffsets, receiveOffsets;
  bench::DeviceBuffer<Element> source, destination;
  bench::DeviceBuffer<size_t> deviceSends, deviceReceives;

  Case(bench::PurlinRuntime& runtime, Operation operation, Pattern pattern,
       const char* label, size_t bytes, int owner)
    : op(operation), name(label), total(bytes),
      layout(makeLayout(op, pattern, bytes, owner, runtime.rank, runtime.world)),
      sendOffsets(bench::offsets(layout.sends)), receiveOffsets(bench::offsets(layout.receives)),
      // Zero-sized buffers deliberately have null pointers.
      source(bench::totalBytes(layout.sends) / sizeof(Element), runtime.stream),
      destination(bench::totalBytes(layout.receives) / sizeof(Element), runtime.stream),
      deviceSends(runtime.world, runtime.stream), deviceReceives(runtime.world, runtime.stream) {
    CHECK_CUDA(cudaMemcpyAsync(deviceSends.get(), layout.sends.data(), runtime.world * sizeof(size_t),
                              cudaMemcpyHostToDevice, runtime.stream));
    CHECK_CUDA(cudaMemcpyAsync(deviceReceives.get(), layout.receives.data(), runtime.world * sizeof(size_t),
                              cudaMemcpyHostToDevice, runtime.stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime.stream));
  }

  void enqueue(bench::PurlinRuntime& runtime, unsigned* iteration, int* errors) {
    advanceIteration<<<1, 1, 0, runtime.stream>>>(iteration);
    for (int peer = 0; peer < runtime.world; ++peer) {
      if (layout.sends[peer] != 0) {
        fill<<<32, 256, 0, runtime.stream>>>(source.get() + sendOffsets[peer] / sizeof(Element),
          layout.sends[peer] / sizeof(Element), runtime.rank, op == Operation::agv ? 0 : peer, iteration);
      }
    }
    const auto* src = reinterpret_cast<const cuda::std::byte*>(source.get());
    auto* dst = reinterpret_cast<cuda::std::byte*>(destination.get());
    if (op == Operation::agv) {
      runtime.context.vState = bench::makePurlinVState(layout.receives, receiveOffsets, runtime.rank);
      purlin::allGatherV<ARCH>(src, dst, deviceReceives.get(), runtime.context, runtime.stream);
    } else if (op == Operation::rsv) {
      runtime.context.vState = bench::makePurlinVState(layout.sends, sendOffsets, runtime.rank);
      purlin::reduceScatterV<ARCH, Element>(src, dst, deviceSends.get(), runtime.context, runtime.stream);
    } else {
      runtime.context.vState = bench::makePurlinAllToAllVState(
        layout.sends, layout.receives, sendOffsets, runtime.rank);
      purlin::all2allV<ARCH>(src, dst, deviceSends.get(), deviceReceives.get(), runtime.context, runtime.stream);
    }
    for (int peer = 0; peer < runtime.world; ++peer) {
      if (layout.receives[peer] != 0) {
        check<<<32, 256, 0, runtime.stream>>>(destination.get() + receiveOffsets[peer] / sizeof(Element),
          layout.receives[peer] / sizeof(Element), peer, op == Operation::agv ? 0 : runtime.rank,
          runtime.world, op == Operation::rsv, iteration, errors);
      }
    }
    CHECK_CUDA(cudaGetLastError());
  }
};

void exercise(bench::PurlinRuntime& runtime, const std::vector<Case*>& sequence,
              int owner, bool graphMode) {
  const int iterations = sequence.size() == 1 ? 32 : 8;
  constexpr int replays = 4;
  bench::DeviceBuffer<int> errors(1, runtime.stream);
  bench::DeviceBuffer<unsigned> iteration(1, runtime.stream);
  CHECK_CUDA(cudaMemsetAsync(errors.get(), 0, sizeof(int), runtime.stream));
  CHECK_CUDA(cudaMemsetAsync(iteration.get(), 0, sizeof(unsigned), runtime.stream));
  CHECK_CUDA(cudaStreamSynchronize(runtime.stream));
  const bool mixed = sequence.size() != 1;
  const char* name = mixed ? "mixed-varlen" : sequence.front()->name;
  const size_t total = mixed ? 0 : sequence.front()->total;
  if (runtime.rank == 0) {
    std::printf("RUN %s: total=%zu owner=%d mode=%s\n", name, total, owner,
                graphMode ? "graph" : "stream");
    std::fflush(stdout);
  }

  const auto enqueueBatch = [&] {
    // Delay the nonempty owner. Three unpaced calls suffice to wrap the two
    // slots; there is no host/rank synchronization between calls or replays.
    if (runtime.rank == owner) delayRank<<<1, 1, 0, runtime.stream>>>();
    for (int i = 0; i < iterations; ++i) {
      for (Case* test : sequence) test->enqueue(runtime, iteration.get(), errors.get());
    }
  };
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (graphMode) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CHECK_CUDA(cudaStreamBeginCapture(runtime.stream, cudaStreamCaptureModeGlobal));
    enqueueBatch();
    CHECK_CUDA(cudaStreamEndCapture(runtime.stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < replays; ++replay) {
      CHECK_CUDA(cudaGraphLaunch(executable, runtime.stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(runtime.stream));
    CHECK_CUDA(cudaGraphExecDestroy(executable));
    CHECK_CUDA(cudaGraphDestroy(graph));
  } else {
    for (int replay = 0; replay < replays; ++replay) enqueueBatch();
    CHECK_CUDA(cudaStreamSynchronize(runtime.stream));
  }
  int localErrors = 0, maximumErrors = 0;
  CHECK_CUDA(cudaMemcpy(&localErrors, errors.get(), sizeof(int), cudaMemcpyDeviceToHost));
  MPI_CHECK(MPI_Allreduce(&localErrors, &maximumErrors, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  if (maximumErrors != 0) throw std::runtime_error(std::string(name) + " returned incorrect data");
  if (runtime.rank == 0) {
    std::printf("PASS %s: total=%zu owner=%d mode=%s calls=%zu\n", name, total, owner,
                graphMode ? "graph" : "stream", iterations * replays * sequence.size());
    std::fflush(stdout);
  }
}
}

int main() {
  try {
    bench::PurlinRuntime runtime;
    for (int owner = 0; owner < runtime.world; ++owner) {
      std::vector<std::unique_ptr<Case>> cases;
      std::vector<Case*> mixed;
      for (size_t total : {128, 8192, 1 << 20, 16 << 20}) {
        const auto add = [&](Operation op, Pattern pattern, const char* name) {
          cases.push_back(std::make_unique<Case>(runtime, op, pattern, name, total, owner));
          mixed.push_back(cases.back().get());
        };
        add(Operation::agv, Pattern::oneWay, "AGV");
        add(Operation::rsv, Pattern::oneWay, "RSV");
        add(Operation::a2av, Pattern::oneWay, "A2AV-one-way");
        add(Operation::a2av, Pattern::selfOnly, "A2AV-self-only");
        add(Operation::a2av, Pattern::mixedRing, "A2AV-mixed-ring");
      }
      for (bool graphMode : {false, true}) {
        for (const auto& test : cases) exercise(runtime, {test.get()}, owner, graphMode);
        // Share context, epochs, signals and staging across collective kinds,
        // grid sizes, and packet/nonchunked/chunked dispatches in one sequence.
        exercise(runtime, mixed, owner, graphMode);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    return bench::reportFailure(error);
  }
}
