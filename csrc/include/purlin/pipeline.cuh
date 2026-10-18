#ifndef PURLIN_PIPELINE_CUH
#define PURLIN_PIPELINE_CUH

#include <cuda/std/cstddef>

#include "static_for.cuh"

namespace purlin {
  // Compile-time interface: derived operations supply Value and implement every
  // method below. Deleted defaults catch missing methods at compile-time without requiring virtual dispatch.
  struct PipelineOperation {
    // Resolve the input for a runtime or compile-time stage index.
    template<typename Stage>
    __device__ auto prepare(Stage) const = delete;

    // Advance consumption state once per completed stage, before waiting.
    __device__ void advance() = delete;

    // Prefetch one value into the calling thread's shared-memory slot.
    template<typename Value, typename Input, typename ElementIdx>
    __device__ void prefetch(Value* const&, const Input&, ElementIdx) const = delete;

    // Commit the current prefetch group before consuming the completed stage.
    __device__ void commit() const = delete;

    // Wait until at most N committed prefetch groups remain outstanding.
    template<int N>
    __device__ void wait() const = delete;

    // Consume the ready array of Config::ELEMS_PER_THREAD values.
    template<typename Values>
    __device__ void consume(int completedStage, const Values&) = delete;
  };

  template<typename Config>
  struct Pipeline {
    template<typename Operation>
    __device__ __forceinline__
    static void run(cuda::std::byte* __restrict__ const& workspace,
      const int totalStages, Operation& operation, const unsigned int id = threadIdx.x) {
      static_assert(cuda::std::is_base_of_v<PipelineOperation, Operation>,
        "pipeline operations must implement PipelineOperation");
      using Value = typename Operation::Value;
      constexpr int stageElements = Config::ELEMS_PER_THREAD * Config::THREADS;
      auto* __restrict__ pipeline = reinterpret_cast<Value*>(workspace);
      // id is this thread's index within the group of Config::THREADS participants.

      // prime
      purlin::static_for<Config::PIPE_STAGES>([&](auto globalStage) {
        const auto input = operation.prepare(globalStage);
        auto* __restrict__ stage = pipeline + globalStage * stageElements;
        purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          const auto slot = i * Config::THREADS + id;
          operation.prefetch(stage + slot, input, i);
        });
        operation.commit();
      });

      Value values[Config::ELEMS_PER_THREAD];
      // steady state
      for (int globalStage = Config::PIPE_STAGES; globalStage < totalStages; ++globalStage) {
        operation.advance();
        const auto completedStage = globalStage - Config::PIPE_STAGES;
        const auto circularStage = globalStage % Config::PIPE_STAGES;
        const auto input = operation.prepare(globalStage);
        auto* __restrict__ stage = pipeline + circularStage * stageElements;
        operation.template wait<Config::PIPE_STAGES - 1>();
        purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          const auto slot = i * Config::THREADS + id;
          values[i] = stage[slot];
          operation.prefetch(stage + slot, input, i);
        });
        operation.commit();
        operation.consume(completedStage, values);
      }

      // drain
      purlin::static_for<Config::PIPE_STAGES>([&](auto remaining) {
        operation.advance();
        const auto completedStage = totalStages - Config::PIPE_STAGES + remaining;
        const auto circularStage = completedStage % Config::PIPE_STAGES;
        auto* __restrict__ stage = pipeline + circularStage * stageElements;
        operation.template wait<Config::PIPE_STAGES - 1 - remaining>();
        purlin::static_for<Config::ELEMS_PER_THREAD>([&](auto i) {
          const auto slot = i * Config::THREADS + id;
          values[i] = stage[slot];
        });
        operation.consume(completedStage, values);
      });
    }
  };
}

#endif // PURLIN_PIPELINE_CUH
