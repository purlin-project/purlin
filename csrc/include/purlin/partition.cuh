//
// Created by osayamen on 5/25/26.
//

#ifndef PURLIN_PARTITION_CUH
#define PURLIN_PARTITION_CUH

#include "static_for.cuh"
#include <cub/cub.cuh>
#include "constants.cuh"
namespace purlin {
  __host__ __device__ __forceinline__
  constexpr size_t alignUp(const size_t offset, const size_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
  }

  struct PartitionResult {
    const size_t bytes;
    const size_t startOffset;
  };
  struct PeerBlock {
    int peer;
    const int intraIdx;
    const int blockSetSize;
  };
  __host__ __device__ __forceinline__
  constexpr size_t getWeightedPeerBlockStateBytes() {
    using ByteReduce = cub::WarpReduce<size_t>;
    using IntReduce = cub::WarpReduce<int>;
    size_t offset = 0;
    offset = alignUp(offset, alignof(size_t));
    offset += MAX_RANKS_PER_DOMAIN * sizeof(size_t);
    offset = alignUp(offset, alignof(size_t));
    offset += sizeof(size_t);
    offset = alignUp(offset, alignof(int));
    offset += MAX_RANKS_PER_DOMAIN * sizeof(int);
    offset = alignUp(offset, alignof(int));
    offset += MAX_RANKS_PER_DOMAIN * sizeof(int);
    offset = alignUp(offset, alignof(int));
    offset += sizeof(int);
    offset = alignUp(offset, alignof(typename ByteReduce::TempStorage));
    offset += sizeof(typename ByteReduce::TempStorage);
    offset = alignUp(offset, alignof(typename IntReduce::TempStorage));
    offset += sizeof(typename IntReduce::TempStorage);
    return cuda::std::bit_ceil(static_cast<uint32_t>(offset));
  }
  constexpr auto WEIGHTED_PEER_BLOCK_STATE_BYTES = getWeightedPeerBlockStateBytes();

  __device__ __forceinline__
  static auto mapPeerBlock(const int bIdx, const int blockSetSize) {
    return PeerBlock{
      .peer = bIdx / blockSetSize,
      .intraIdx = bIdx % blockSetSize,
      .blockSetSize = blockSetSize,
    };
  }
  template<typename WT>
  __device__ __forceinline__
  static auto mapPeerBlock(const int bIdx, const int blockSetSize, const int rank, const WT& world) {
    static_assert(cuda::std::is_same_v<WT, int> || cuda::std::is_same_v<WT, cuda::fast_mod_div<int, true>>);
    const auto peer = ((bIdx / blockSetSize) + rank + 1) % world;
    return PeerBlock{
      .peer = peer,
      .intraIdx = bIdx % blockSetSize,
      .blockSetSize = blockSetSize,
    };
  }

  // Uneven per-peer split for worlds that do not divide the block count: the
  // first (blocks % world) peers take one extra block, so every block maps to a
  // real peer and no block idles. Divisible worlds reduce to the uniform
  // mapping. Requires blocks >= world.
  template<typename WT>
  __device__ __forceinline__
  static auto mapPeerBlockUneven(const int bIdx, const int blocks, const WT& world) {
    static_assert(cuda::std::is_same_v<WT, int> || cuda::std::is_same_v<WT, cuda::fast_mod_div<int, true>>);
    const int base = blocks / world;
    const int rem = blocks % world;
    const int pivot = rem * (base + 1);
    if (bIdx < pivot) {
      return PeerBlock{
        .peer = bIdx / (base + 1),
        .intraIdx = bIdx % (base + 1),
        .blockSetSize = base + 1,
      };
    }
    const int shifted = bIdx - pivot;
    return PeerBlock{
      .peer = rem + shifted / base,
      .intraIdx = shifted % base,
      .blockSetSize = base,
    };
  }

  template<typename WT>
  __device__ __forceinline__
  constexpr auto isSkewed(const size_t& totalBytes, const size_t& maxBytes, const WT& world) {
    return maxBytes >= 2 * (totalBytes / world);
  }
  template<typename WT>
  __device__ __forceinline__
  static auto mapWeightedPeerBlock(const int bIdx, const int consumerBlocks,
    const size_t* __restrict__ const& sizes,
    cuda::std::byte* __restrict__ const& workspace,
    const WT& world) {
    static_assert(cuda::std::is_same_v<WT, int> || cuda::std::is_same_v<WT, cuda::fast_mod_div<int, true>>);
    using ByteReduce = cub::WarpReduce<size_t>;
    using IntReduce = cub::WarpReduce<int>;
    auto* __restrict__ remainders = reinterpret_cast<size_t*>(workspace);
    auto* __restrict__ totalBytes = remainders + MAX_RANKS_PER_DOMAIN;
    auto* __restrict__ blockCounts = reinterpret_cast<int*>(totalBytes + 1);
    auto* __restrict__ blockOffsets = blockCounts + MAX_RANKS_PER_DOMAIN;
    auto* __restrict__ nonzeroPeers = blockOffsets + MAX_RANKS_PER_DOMAIN;
    auto* __restrict__ byteReduceStorage =
      reinterpret_cast<typename ByteReduce::TempStorage*>(nonzeroPeers + 1);
    auto* __restrict__ intReduceStorage =
      reinterpret_cast<typename IntReduce::TempStorage*>(byteReduceStorage + 1);
    const int worldI = world;
    const auto laneId = static_cast<int>(threadIdx.x % WARP_SIZE);
    const auto warpId = static_cast<int>(threadIdx.x / WARP_SIZE);
    if (warpId == 0) {
      size_t localBytes = 0;
      int localNonzeroPeers = 0;
      for (int peer = laneId; peer < worldI; peer += WARP_SIZE) {
        const auto peerBytes = sizes[peer];
        localBytes += peerBytes;
        localNonzeroPeers += peerBytes > 0;
        blockCounts[peer] = 0;
        blockOffsets[peer] = 0;
        remainders[peer] = 0;
      }
      const auto totalBytesReduced = ByteReduce(*byteReduceStorage).Sum(localBytes);
      const auto nonzeroPeersReduced = IntReduce(*intReduceStorage).Sum(localNonzeroPeers);
      if (!laneId) {
        *totalBytes = totalBytesReduced;
        *nonzeroPeers = nonzeroPeersReduced;
      }
      __syncwarp();

      if (*totalBytes == 0 || consumerBlocks <= 0) {
        if (!laneId) {
          blockCounts[0] = consumerBlocks;
        }
      }
      else if (consumerBlocks >= *nonzeroPeers) {
        const auto remainingBlocks = consumerBlocks - *nonzeroPeers;
        int localBlocks = 0;
        for (int peer = laneId; peer < worldI; peer += WARP_SIZE) {
          const auto peerBytes = sizes[peer];
          if (peerBytes > 0) {
            const auto weightedBlocks = static_cast<size_t>(remainingBlocks) * peerBytes;
            const auto extraBlocks = static_cast<int>(weightedBlocks / *totalBytes);
            const auto peerBlocks = 1 + extraBlocks;
            localBlocks += peerBlocks;
            blockCounts[peer] = peerBlocks;
            remainders[peer] = weightedBlocks - static_cast<size_t>(extraBlocks) * *totalBytes;
          }
        }
        const auto assignedBlocks = IntReduce(*intReduceStorage).Sum(localBlocks);
        __syncwarp();
        for (int leftover = consumerBlocks - assignedBlocks; !laneId && leftover > 0; --leftover) {
          int bestPeer = -1;
          size_t bestRemainder = 0;
          for (int peer = 0; peer < worldI; ++peer) {
            if (sizes[peer] > 0 && (bestPeer < 0 || remainders[peer] > bestRemainder)) {
              bestPeer = peer;
              bestRemainder = remainders[peer];
            }
          }
          blockCounts[bestPeer]++;
          remainders[bestPeer] = 0;
        }
      }
      else {
        for (int block = 0; !laneId && block < consumerBlocks; ++block) {
          int bestPeer = -1;
          size_t bestBytes = 0;
          for (int peer = 0; peer < worldI; ++peer) {
            if (!blockCounts[peer] && sizes[peer] > bestBytes) {
              bestPeer = peer;
              bestBytes = sizes[peer];
            }
          }
          blockCounts[bestPeer] = 1;
        }
      }

      __syncwarp();
      for (int peerIdx = laneId; peerIdx < worldI; peerIdx += WARP_SIZE) {
        int blockOffset = 0;
        for (int peer = 0; peer < peerIdx; ++peer) {
          blockOffset += blockCounts[peer];
        }
        blockOffsets[peerIdx] = blockOffset;
      }
    }
    __syncthreads();

    int peer = 0;
    int intraIdx = bIdx;
    int blockSetSize = blockCounts[0];
    for (int p = 0; p < worldI; ++p) {
      const auto blockOffset = blockOffsets[p];
      const auto peerBlocks = blockCounts[p];
      if (bIdx >= blockOffset && bIdx < blockOffset + peerBlocks) {
        peer = p;
        intraIdx = bIdx - blockOffset;
        blockSetSize = peerBlocks;
        break;
      }
    }
    __syncthreads();
    return PeerBlock{
      .peer = peer,
      .intraIdx = intraIdx,
      .blockSetSize = blockSetSize,
    };
  }

  // Block slices are cut at the sector grain, so that when the buffer base is
  // sector-aligned every block's slice base is too. Slicing at the access
  // alignment alone leaves a ragged (variable-size) buffer's slices starting
  // mid-sector, and every line of a long remote copy then straddles two
  // sectors: measured at 16-24% on the zero-staged variable-size collectives,
  // and the same on the staged ones wherever they run non-chunked. Whole
  // sectors are spread evenly and the sub-sector tail, still a multiple of the
  // access alignment, rides with the last block so every slice is contiguous.
  // A transfer too small to give every block a sector keeps the fine grain, so
  // no block idles and small transfers slice exactly as before. A caller that
  // already slices coarser than a sector keeps its own grain.
  constexpr int PARTITION_GRAIN_BYTES = 128;

  // One grain, fixed at compile time, so every multiply and modulo is a shift
  // or a mask. The sub-grain tail (a multiple of AlignmentBytes) rides with the
  // last block; a caller that knows there is no tail skips that work.
  template<int Grain, int AlignmentBytes, bool tailed, typename BT>
  __device__ __forceinline__
  constexpr auto partitionAt(const size_t& aligned, const BT& blocks, const int& bIdx) {
    static_assert(Grain % AlignmentBytes == 0, "the grain must be a multiple of the access alignment");
    const long int units = static_cast<long int>(aligned / Grain);
    const auto baseUnits = static_cast<size_t>(units / blocks);
    const auto residue = static_cast<int>(units % blocks);
    const auto ctaUnits = baseUnits + (bIdx < residue);
    const auto offsetUnits = baseUnits * bIdx + cuda::std::min(bIdx, residue);
    size_t bytesSliced = ctaUnits * Grain;
    if constexpr (tailed) {
      if (bIdx == static_cast<int>(blocks) - 1) {
        bytesSliced += aligned % Grain;
      }
    }
    return PartitionResult{
      .bytes = bytesSliced,
      .startOffset = offsetUnits * Grain,
    };
  }

  template<int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const BT& blocks, const int& bIdx) {
    static_assert(cuda::std::is_integral_v<BT> || cuda::std::is_same_v<cuda::fast_mod_div<long int>, BT>);
    constexpr int sectorGrain = AlignmentBytes > PARTITION_GRAIN_BYTES ? AlignmentBytes : PARTITION_GRAIN_BYTES;
    // Bytes below the access alignment were never copied; keep that contract.
    const size_t aligned = (bytes / AlignmentBytes) * AlignmentBytes;
    if (aligned / sectorGrain >= static_cast<size_t>(static_cast<int>(blocks))) {
      return partitionAt<sectorGrain, AlignmentBytes, true>(aligned, blocks, bIdx);
    }
    return partitionAt<AlignmentBytes, AlignmentBytes, false>(aligned, blocks, bIdx);
  }
  // A compile-time byte count that spans at least MAX_NUM_CTAS sectors gives
  // every legal grid a sector per block, so the sector test folds away and the
  // tail is known at compile time (zero for a power-of-two chunk).
  template<size_t bytes, int AlignmentBytes, typename BT = int>
  __device__ __forceinline__
  constexpr auto partition(const BT& blocks, const int& bIdx) {
    constexpr int sectorGrain = AlignmentBytes > PARTITION_GRAIN_BYTES ? AlignmentBytes : PARTITION_GRAIN_BYTES;
    constexpr size_t aligned = (bytes / AlignmentBytes) * AlignmentBytes;
    if constexpr (aligned / sectorGrain >= MAX_NUM_CTAS) {
      return partitionAt<sectorGrain, AlignmentBytes, (aligned % sectorGrain) != 0>(aligned, blocks, bIdx);
    } else {
      return partition<AlignmentBytes>(bytes, blocks, bIdx);
    }
  }
  template<size_t bytes, int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const int& bIdx) {
    return partition<bytes, AlignmentBytes>(blocks, bIdx);
  }
  template<int blocks, int AlignmentBytes>
  __device__ __forceinline__
  constexpr auto partition(const size_t& bytes, const int& bIdx) {
    return partition<AlignmentBytes>(bytes, blocks, bIdx);
  }

  // Scratch bytes prefixSum requires in its workspace argument.
  template<int threads>
  __host__ __device__ __forceinline__
  constexpr size_t prefixSumScratchBytes() {
    if constexpr (MAX_RANKS_PER_DOMAIN > WARP_SIZE) {
      using BlockScan = cub::BlockScan<size_t, threads, cub::BLOCK_SCAN_WARP_SCANS>;
      return sizeof(typename BlockScan::TempStorage);
    } else {
      return sizeof(typename cub::WarpScan<size_t>::TempStorage);
    }
  }

  template<int threads>
  __device__ __forceinline__
  auto prefixSum(const size_t* __restrict__ const& inputs,
    size_t* __restrict__ const& offsets,
    cuda::std::byte* __restrict__ const& workspace,
    const int& n) {
    if constexpr (MAX_RANKS_PER_DOMAIN > WARP_SIZE) {
      constexpr int elems = cuda::ceil_div(MAX_RANKS_PER_DOMAIN, threads);
      static_assert(elems <= 8);
      using BlockScan = cub::BlockScan<size_t, threads, cub::BLOCK_SCAN_WARP_SCANS>;
      auto* __restrict__ scanStorage = reinterpret_cast<typename BlockScan::TempStorage*>(workspace);
      size_t vals[elems];
      purlin::static_for<elems>([&](auto i) {
        const auto idx = i * threads + threadIdx.x;
        if (idx < n) {
          vals[i] = inputs[idx];
        }
        else {
          vals[i] = 0;
        }
      });
      BlockScan(*scanStorage).ExclusiveSum(vals, vals);
      purlin::static_for<elems>([&](auto i) {
        const auto idx = i * threads + threadIdx.x;
        if (idx < n) {
          offsets[idx] = vals[i];
        }
      });
    }
    else {
      const auto warpId = threadIdx.x / WARP_SIZE;
      const auto laneId = threadIdx.x % WARP_SIZE;
      if (warpId == 0) {
        using WarpScan = cub::WarpScan<size_t>;
        auto* __restrict__ scanStorage = reinterpret_cast<typename WarpScan::TempStorage*>(workspace);
        size_t val = 0;
        if (laneId < n) {
          val = inputs[laneId];
        }
        WarpScan(*scanStorage).ExclusiveSum(val, val);
        if (laneId < n) {
          offsets[laneId] = val;
        }
      }
    }
    __syncthreads();
  }
}
#endif //PURLIN_PARTITION_CUH
