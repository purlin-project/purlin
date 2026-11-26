//
// Created by Osayamen on 4/7/26.
//

#ifndef PURLIN_FASCIA_CUH
#define PURLIN_FASCIA_CUH
#include "base.cuh"
#include "pipeline.cuh"
#include "static_for.cuh"

namespace purlin::fascia {
  template<typename Config, typename Dest, typename Src>
  struct CopyOperation : VectorPipelineOp {
    using Element = typename AlignedType<Config::ALIGNMENT_BYTES>::type;
    using Value = AlignedArray<Element, Config::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(Element)>;

    struct State {};
    const Dest& dst;
    const Src& src;

    template<int Vectors>
    __device__ __forceinline__
    State beginGroup() const { return {}; }

    template<int BatchSize>
    __device__ __forceinline__
    void consume(State&, int, size_t index, int firstSource) const {
      Value values[BatchSize];
      purlin::static_for<BatchSize>([&](auto p) {
        const auto* __restrict__ source = reinterpret_cast<const Value*>(src.advance(firstSource + p));
        values[p] = purlin::load(source + index);
      });
      purlin::static_for<BatchSize>([&](auto p) {
        auto* __restrict__ destination = reinterpret_cast<Value*>(dst.advance(firstSource + p));
        purlin::store(destination + index, values[p]);
      });
    }

    template<int Vectors>
    __device__ __forceinline__
    void finishGroup(const State&, size_t) const {}
  };

  template<typename Config, typename RedOp, typename Element>
  struct ReduceOperation : VectorPipelineOp {
    using VE = cuda::std::conditional_t<
      (Config::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<Element>::type, Element>;
    using AccumType = cuda::std::conditional_t<
      (Config::GMEM_ACCESS_ALIGNMENT_BYTES > sizeof(Element)), typename PackedElement<ReduceAccumType<Element>>::type,
      ReduceAccumType<Element>>;
    using VERaw = typename DataToRawType<VE>::type;
    static constexpr int vectorWidth = Config::GMEM_ACCESS_ALIGNMENT_BYTES / sizeof(VE);
    using Accumulator = AlignedArray<AccumType, vectorWidth>;
    using Value = AlignedArray<VERaw, vectorWidth>;
    static_assert(cuda::std::is_trivially_copyable_v<Value>);

    template<int Vectors>
    using State = AlignedArray<Accumulator, Vectors, alignof(Accumulator)>;

    const ReduceTRArgs& args;
    const RedOp reduceOp{};

    template<int Vectors>
    __device__ __forceinline__
    State<Vectors> beginGroup() const {
      State<Vectors> accumulators;
      constexpr typename RedOp::template Identity<AccumType> clear{};
      purlin::static_for<Vectors>([&](auto slot) {
        purlin::static_for<vectorWidth>([&](auto lane) {
          clear(accumulators[slot][lane]);
        });
      });
      return accumulators;
    }

    template<int BatchSize, int Vectors>
    __device__ __forceinline__
    void consume(State<Vectors>& accumulators, int slot, size_t index, int firstSource) const {
      constexpr Converter<AccumType, VE> loadConv{};
      Value rawValues[BatchSize];
      Accumulator values[BatchSize];
      purlin::static_for<BatchSize>([&](auto p) {
        const auto* __restrict__ source = reinterpret_cast<const Value*>(
          args.sources[firstSource + p] + args.residualOffset);
        rawValues[p] = source[index];
      });
      purlin::static_for<BatchSize>([&](auto p) {
        purlin::static_for<vectorWidth>([&](auto lane) {
          values[p][lane] = loadConv(rawValues[p][lane]);
        });
      });
      // Source batches and the values within each batch are visited in rank order.
      purlin::static_for<BatchSize>([&](auto p) {
        reduceOp(accumulators[slot], values[p]);
      });
    }

    template<int Vectors>
    __device__ __forceinline__
    void finishGroup(const State<Vectors>& accumulators, size_t firstVector) const {
      constexpr Converter<VERaw, AccumType> storeConv{};
      auto* __restrict__ destination = reinterpret_cast<Value*>(args.dst);
      purlin::static_for<Vectors>([&](auto slot) {
        Value result{};
        purlin::static_for<vectorWidth>([&](auto lane) {
          result[lane] = storeConv(accumulators[slot][lane]);
        });
        destination[firstVector + static_cast<size_t>(slot) * Config::THREADS] = result;
      });
    }
  };
}

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

  template<
    typename Dest,
    typename Src,
    typename Workspace>
  requires ((cuda::std::is_base_of_v<CopyArgs, Dest> || cuda::std::is_convertible_v<const Dest&, cuda::std::byte*>)
    && (cuda::std::is_base_of_v<CopyArgs, Src> || cuda::std::is_convertible_v<const Src&, const cuda::std::byte*>))
  __device__ __forceinline__
  static void copy(const Dest& dst, const Src& src, const size_t& bytes,
    const Workspace& /*workspace is not needed*/) {
    const auto& dstArgs = asCopyArgs<cuda::std::byte*>(dst);
    const auto& srcArgs = asCopyArgs<const cuda::std::byte*>(src);
    using Operation = fascia::CopyOperation<
      Config,
      cuda::std::remove_cvref_t<decltype(dstArgs)>,
      cuda::std::remove_cvref_t<decltype(srcArgs)>
    >;
    const Operation operation{.dst = dstArgs, .src = srcArgs};
    VectorPipeline<Config>::run(bytes / GMEM_ACCESS_ALIGNMENT_BYTES, srcArgs.n, operation);
  }

  template<
    ReduceResult result,
    ReduceOp ro = ReduceOp::add,
    typename RedOp = LoweredReduceOp<ro, NARCH>::type,
    typename Element
  >
  __device__ __forceinline__
  static void reduce(const ReduceTRArgs& redArgs, Element* __restrict__ const&) {
    static_assert(result == ReduceResult::unicast);
    const fascia::ReduceOperation<Config, RedOp, Element> operation{.args = redArgs};
    VectorPipeline<Config>::run(redArgs.bytesRed / GMEM_ACCESS_ALIGNMENT_BYTES, redArgs.world, operation);
  }
};
#endif //PURLIN_FASCIA_CUH
