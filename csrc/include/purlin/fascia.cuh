//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_FASCIA_CUH
#define PURLIN_FASCIA_CUH
#include "static_for.cuh"
#include "base.cuh"
template<typename Cfg_>
struct purlin::Atom<700, Cfg_> {
  static_assert(Cfg_::MEMTYPE == MemType::unicast, "the multimem datapath requires sm90 or newer");
  using BaseConfig = Cfg_;
  using Config = Cfg_;
  static constexpr int NARCH = 700;
  static constexpr int COPY_PIPELINE_BYTES = 0;
  static constexpr int RED_PIPELINE_BYTES = COPY_PIPELINE_BYTES;
  static constexpr int COPY_PIPELINE_SMEM_BYTES = 0;
  static constexpr int RED_PIPELINE_SMEM_BYTES = COPY_PIPELINE_SMEM_BYTES;
  static constexpr int THREADS = Config::THREADS;
  static constexpr int WARPS = THREADS / WARP_SIZE;
  static constexpr int STAGE_BYTES = 0;
  static constexpr int GMEM_ACCESS_ALIGNMENT_BYTES = Config::GMEM_ACCESS_ALIGNMENT_BYTES;

  __device__ __forceinline__
  static void copy(cuda::std::byte* __restrict__ const& dst,
    const cuda::std::byte* __restrict__ const& src,
    const size_t& bytes,
    const cuda::std::byte* __restrict__ const& /*workspace is not needed*/) {
    using Element = AlignedType<Config::ALIGNMENT_BYTES>::type;
    constexpr int vectorWidth = Config::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(Element);
    using VT = AlignedArray<Element, vectorWidth>;
    using IndexT = uint32_t;
    const auto tIdx = threadIdx.x;
    const auto vP = static_cast<IndexT>(bytes / Config::GMEM_ACCESS_ALIGNMENT_BYTES);
    auto* __restrict__ vD = reinterpret_cast<VT*>(dst);
    const auto* __restrict__ vS = reinterpret_cast<const VT*>(src);
    const auto threadElems = vP / Config::THREADS;
    const auto trips = threadElems / Config::UNROLL_FACTOR;
    for (int i = 0; i < trips; ++i) {
      VT reginald[Config::UNROLL_FACTOR];
      IndexT indices[Config::UNROLL_FACTOR];
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + tIdx;
      });
      // Load the source values from global memory into registers.
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        reginald[j] = vS[indices[j]];
      });
      // Write registers to the destination.
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        purlin::store(vD + indices[j], reginald[j]);
      });
    }
    const auto cutoff = trips * Config::UNROLL_FACTOR * Config::THREADS;
    if (vP > cutoff) {
      const auto residue = vP - cutoff;
      vS += cutoff;
      vD += cutoff;
      for (int i = static_cast<int>(tIdx); i < residue; i += Config::THREADS) {
        purlin::store(vD + i, vS[i]);
      }
    }
  }

  template<ReduceResult result, ReduceOp ro = ReduceOp::add,
    typename RedOp = typename LoweredReduceOp<ro, NARCH>::type, typename Element>
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const&) {
    static_assert(result == ReduceResult::unicast,
      "this datapath stores reduction results with unicast writes only");
    // Generic register reducer. Read every peer's replica in ascending rank
    // order, accumulate in f32, and store the converted result.
    constexpr RedOp op{};
    using VE = cuda::std::conditional_t<
      (Config::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
    ReduceAccumType<Element>>;
    using VERaw = DataToRawType<VE>::type;
    constexpr int vectorWidth = Config::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(VE);
    using AVT = AlignedArray<AccumType, vectorWidth>;
    using LVT = AlignedArray<VERaw, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<LVT>);
    constexpr Converter<AccumType, VE> loadConv{};
    constexpr Converter<VERaw, AccumType> storeConv{};
    auto* __restrict__ vD = reinterpret_cast<LVT*>(redArgs.dst);
    const auto redElems = redArgs.bytesRed / Config::GMEM_ACCESS_ALIGNMENT_BYTES;
    const auto threadElems = redElems / Config::THREADS;
    const auto trips = threadElems / Config::UNROLL_FACTOR;
    const auto worldTrips = redArgs.world / Config::WORLD_UNROLL;
    AVT accumulators[Config::UNROLL_FACTOR];
    constexpr typename RedOp::template Identity<AccumType> clear{};
    const auto cutoff = worldTrips * Config::WORLD_UNROLL;
    purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
      purlin::static_for<vectorWidth>([&](auto k) {
        clear(accumulators[j][k]);
      });
    });
    for (int i = 0; i < trips; ++i) {
      uint indices[Config::UNROLL_FACTOR];
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        indices[j] = (i * Config::UNROLL_FACTOR + j) * Config::THREADS + threadIdx.x;
      });
      // Reduce this group of elements across all ranks.
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        // Visit ranks in ascending order so floating-point reductions are
        // deterministic.
        for (int t = 0; t < worldTrips; ++t) {
          LVT wendell[Config::WORLD_UNROLL];
          AVT arnold[Config::WORLD_UNROLL];
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + redArgs.residualOffset);
            // Load this rank's values from global memory into registers.
            wendell[p] = vData[indices[j]];
          });
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            AVT val{};
            const auto valRaw = wendell[p];
            purlin::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            arnold[p] = val;
          });
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            op(accumulators[j], arnold[p]);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = worldTrips * Config::WORLD_UNROLL; peer < redArgs.world; ++peer) {
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + redArgs.residualOffset);
            const auto valRaw = vData[indices[j]];
            AVT val{};
            purlin::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulators[j], val);
          }
        }
      });
      // Convert and store the accumulated results.
      purlin::static_for<Config::UNROLL_FACTOR>([&](auto j) {
        LVT resultRaw{};
        purlin::static_for<resultRaw.size()>([&](auto k) {
            resultRaw[k] = storeConv(accumulators[j][k]);
        });
        vD[indices[j]] = resultRaw;
        purlin::static_for<resultRaw.size()>([&](auto k) {
            clear(accumulators[j][k]);
        });
      });
    }
    const auto redCutoff = static_cast<size_t>(trips) * Config::UNROLL_FACTOR * Config::THREADS;
    if (redElems > redCutoff) {
      vD += redCutoff;
      const auto residue = redElems - redCutoff;
      AVT accumulator{};
      purlin::static_for<accumulator.size()>([&](auto j) {
        clear(accumulator[j]);
      });
      for (int idx = static_cast<int>(threadIdx.x); idx < residue; idx += Config::THREADS) {
        // Reduce the elements that did not fit in a complete unrolled trip.
        for (int t = 0; t < worldTrips; ++t) {
          LVT wendell[Config::WORLD_UNROLL];
          AVT arnold[Config::WORLD_UNROLL];
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            const auto peer = t * Config::WORLD_UNROLL + p;
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + redArgs.residualOffset) + redCutoff;
            // Load this rank's remaining values into registers.
            wendell[p] = vData[idx];
          });
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            AVT val{};
            const auto valRaw = wendell[p];
            purlin::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            arnold[p] = val;
          });
          purlin::static_for<Config::WORLD_UNROLL>([&](auto p) {
            op(accumulator, arnold[p]);
          });
        }
        if (redArgs.world > cutoff) {
          for (int peer = worldTrips * Config::WORLD_UNROLL; peer < redArgs.world; ++peer) {
            auto* __restrict__ vData = reinterpret_cast<const LVT*>(redArgs.sources[peer] + redArgs.residualOffset) + redCutoff;
            const auto valRaw = vData[idx];
            AVT val{};
            purlin::static_for<val.size()>([&](auto k) {
              val[k] = loadConv(valRaw[k]);
            });
            op(accumulator, val);
          }
        }
        // Convert and store the remaining accumulated results.
        LVT resultRaw{};
        purlin::static_for<resultRaw.size()>([&](auto k) {
            resultRaw[k] = storeConv(accumulator[k]);
        });
        vD[idx] = resultRaw;
        purlin::static_for<resultRaw.size()>([&](auto k) {
          clear(accumulator[k]);
        });
      }
    }
  }
};
#endif //PURLIN_FASCIA_CUH
