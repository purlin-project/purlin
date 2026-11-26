#ifndef PURLIN_PIPELINE_CUH
#define PURLIN_PIPELINE_CUH

#include <cuda/std/type_traits>

#include "static_for.cuh"

namespace purlin {
  struct PipelineOperation {
    template<typename Row, typename Stage>
    __device__ void prefetch(Row row, Stage stage) = delete;
    __device__ auto refill(int row, int stage) = delete;
    __device__ auto read(int row) = delete;
    __device__ void advance() = delete;
    __device__ void commit() const = delete;
    // Wait until at most N committed prefetch groups remain outstanding.
    template<int N>
    __device__ void wait() const = delete;
    template<typename Values>
    __device__ void consume(int completedStage, const Values&) = delete;
  };

  template<typename Config>
  struct Pipeline {
    template<typename Operation>
    __device__ __forceinline__
    static void run(int totalStages, Operation& operation) {
      static_assert(cuda::std::is_base_of_v<PipelineOperation, Operation>,
        "pipeline operations must implement PipelineOperation");
      constexpr int depth = Config::PIPE_STAGES;

      // prime
      purlin::static_for<depth>([&](auto stage) {
        operation.prefetch(stage, stage);
        operation.commit();
      });

      // steady state
      for (int stage = depth; stage < totalStages; ++stage) {
        operation.advance();
        operation.template wait<depth - 1>();
        const auto values = operation.refill(stage % depth, stage);
        operation.commit();
        operation.consume(stage - depth, values);
      }

      // drain
      purlin::static_for<depth>([&](auto i) {
        operation.advance();
        const int stage = totalStages - depth + i;
        operation.template wait<depth - 1 - i>();
        const auto values = operation.read(stage % depth);
        operation.consume(stage, values);
      });
    }
  };

  struct VectorPipelineOp {
    __device__ auto beginGroup() = delete;
    template<int BatchSize, typename State>
    __device__ void consume(State& state, int slot, size_t index, int firstSource) = delete;
    template<int Vectors, typename State>
    __device__ void finishGroup(State& state, size_t firstVector) = delete;
  };

  template<typename Config>
  struct VectorPipeline {
    template<typename Operation>
    __device__ __forceinline__
    static void run(size_t vectors, int sourceCount, const Operation& operation) {
      constexpr auto worldUnroll = Config::WORLD_UNROLL;
      constexpr auto groupWidth = static_cast<size_t>(Config::THREADS) * Config::UNROLL_FACTOR;
      const auto trips = vectors / groupWidth;
      const auto sourceCutoff = sourceCount / worldUnroll * worldUnroll;

      const auto runGroup = [&]<int Vectors>(size_t firstVector) {
        auto state = operation.template beginGroup<Vectors>();
        purlin::static_for<Vectors>([&](auto slot) {
          const auto index = firstVector + static_cast<size_t>(slot) * Config::THREADS;
          for (int firstSource = 0; firstSource < sourceCutoff; firstSource += worldUnroll) {
            operation.template consume<worldUnroll>(state, slot, index, firstSource);
          }
          for (int firstSource = sourceCutoff; firstSource < sourceCount; ++firstSource) {
            operation.template consume<1>(state, slot, index, firstSource);
          }
        });
        operation.template finishGroup<Vectors>(state, firstVector);
      };

      #pragma unroll 1
      for (size_t trip = 0; trip < trips; ++trip) {
        runGroup.template operator()<Config::UNROLL_FACTOR>(trip * groupWidth + threadIdx.x);
      }
      #pragma unroll 1
      for (size_t index = trips * groupWidth + threadIdx.x; index < vectors; index += Config::THREADS) {
        runGroup.template operator()<1>(index);
      }
    }
  };
}

#endif // PURLIN_PIPELINE_CUH
